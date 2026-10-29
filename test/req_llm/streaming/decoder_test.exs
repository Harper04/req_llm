defmodule ReqLLM.Streaming.DecoderTest do
  use ExUnit.Case, async: true

  alias ReqLLM.StreamChunk
  alias ReqLLM.Streaming.Decoder

  defmodule StatefulProvider do
    @moduledoc false

    def init_stream_state(model), do: {model.id, 0}

    def parse_stream_protocol(chunk, state),
      do: ReqLLM.Provider.parse_stream_protocol(chunk, state)

    def decode_stream_event(%{data: "terminal"}, _model, state) do
      {[StreamChunk.meta(%{terminal?: true})], state}
    end

    def decode_stream_event(%{data: "string-terminal"}, _model, state) do
      {[StreamChunk.meta(%{"terminal?" => true})], state}
    end

    def decode_stream_event(%{data: text}, _model, {id, count}) when is_binary(text) do
      {[StreamChunk.text(text), StreamChunk.text("#{id}:#{count + 1}")], {id, count + 1}}
    end

    def decode_stream_event(_, _, state), do: {[], state}

    def decode_stream_event(_, _), do: raise("three-arity decoder must take precedence")

    def flush_stream_state(_model, {id, count}) do
      {[StreamChunk.text("flush:#{id}:#{count}")], {id, count + 1}}
    end
  end

  defmodule StatelessProvider do
    @moduledoc false
    def decode_stream_event(%{data: %{"text" => text}}, _model), do: [StreamChunk.text(text)]
    def decode_stream_event(_, _), do: []
  end

  defmodule DefaultProvider do
    @moduledoc false
  end

  defmodule LegacyProvider do
    @moduledoc false

    def init_stream_state(%LLMDB.Model{extra: %{wire: %{protocol: :legacy}}}), do: %{}
  end

  test "canonical streams skip legacy provider initialization" do
    decoder = new_decoder(provider_mod: LegacyProvider, canonical_stream?: true)
    assert {:ok, [], false, _decoder} = Decoder.decode("", decoder)
  end

  test "decodes partial SSE events with no process and preserves multi-chunk order and state" do
    decoder = new_decoder()
    assert {:ok, [], false, decoder} = Decoder.decode("data: o", decoder)
    assert {:ok, chunks, false, decoder} = Decoder.decode("ne\n\ndata: two\n\n", decoder)

    assert chunks == [
             StreamChunk.text("one"),
             StreamChunk.text("test:1"),
             StreamChunk.text("two"),
             StreamChunk.text("test:2")
           ]

    assert {[flush], _decoder} = Decoder.flush_provider(decoder)
    assert flush == StreamChunk.text("flush:test:2")
  end

  test "flush decodes the unfinished SSE event before flushing the updated provider state" do
    assert {:ok, [], false, decoder} = Decoder.decode("data: pending", new_decoder())
    assert {[first, second], false, decoder} = Decoder.flush_protocol(decoder)
    assert first == StreamChunk.text("pending")
    assert second == StreamChunk.text("test:1")
    assert {[flush], decoder} = Decoder.flush_provider(decoder)
    assert flush == StreamChunk.text("flush:test:1")
    assert {[flush], _decoder} = Decoder.flush_provider(decoder)
    assert flush == StreamChunk.text("flush:test:2")
  end

  test "recognizes every raw terminal event form" do
    for data <- [
          "[DONE]",
          %{"done" => true},
          %{"type" => "message_stop"},
          %{"type" => "response.completed"}
        ] do
      parser = fn _, state -> {:ok, [%{data: data}], state} end
      assert {:ok, _, true, _} = Decoder.decode("ignored", new_decoder(protocol_parser: parser))
    end
  end

  test "recognizes atom and string terminal chunk metadata" do
    for data <- ["terminal", "string-terminal"] do
      assert {:ok, [%StreamChunk{type: :meta}], true, _} =
               Decoder.decode("data: #{data}\n\n", new_decoder())
    end
  end

  test "preserves raw-event terminal detection before JSON decoding" do
    assert {:ok, [], false, _} =
             Decoder.decode(~s(data: {"done":true}\n\n), new_decoder())
  end

  test "flush only marks raw terminal events, even when a decoded chunk is terminal" do
    assert {:ok, [], false, decoder} = Decoder.decode("data: terminal", new_decoder())
    assert {[%StreamChunk{type: :meta}], false, _} = Decoder.flush_protocol(decoder)

    assert {:ok, [], false, decoder} = Decoder.decode("data: [DONE]", new_decoder())
    assert {_, true, _} = Decoder.flush_protocol(decoder)
  end

  test "parse errors preserve accumulated bytes and provider state" do
    parser = fn
      "partial", nil -> {:incomplete, :saved}
      "bad", :saved -> {:error, :invalid_frame}
      "rest", :saved -> {:ok, [%{data: "recovered"}], :finished}
    end

    assert {:ok, [], false, decoder} =
             Decoder.decode("partial", new_decoder(protocol_parser: parser))

    assert {:error, :invalid_frame, unchanged} = Decoder.decode("bad", decoder)
    assert unchanged == decoder
    assert {:ok, [content, count], false, decoder} = Decoder.decode("rest", unchanged)
    assert content == StreamChunk.text("recovered")
    assert count == StreamChunk.text("test:1")
    assert {[], false, decoder} = Decoder.flush_protocol(decoder)
    assert {[flush], _} = Decoder.flush_provider(decoder)
    assert flush == StreamChunk.text("flush:test:1")
  end

  test "HTTP fallback discards the custom parser, partial bytes and provider state" do
    parser = fn _, _ -> {:ok, [%{data: "override"}], :custom_state} end

    assert {:ok, [_, _], false, decoder} =
             Decoder.decode("custom", new_decoder(protocol_parser: parser))

    decoder = Decoder.reset_http(decoder)
    assert {:ok, [content, count], false, _} = Decoder.decode("data: http\n\n", decoder)
    assert content == StreamChunk.text("http")
    assert count == StreamChunk.text("test:1")
  end

  test "two-arity decoding receives processed JSON and flushes without provider callbacks" do
    decoder = new_decoder(provider_mod: StatelessProvider)

    assert {:ok, [chunk], false, decoder} =
             Decoder.decode(~s(data: {"text":"hello"}\n\n), decoder)

    assert chunk == StreamChunk.text("hello")
    assert {[], false, decoder} = Decoder.flush_protocol(decoder)
    assert {[], ^decoder} = Decoder.flush_provider(decoder)
  end

  test "missing provider callbacks use the default SSE parser and OpenAI decoder" do
    decoder = new_decoder(provider_mod: DefaultProvider)
    data = ~s(data: {"choices":[{"delta":{"content":"default"}}]}\n\n)
    assert {:ok, [chunk], false, _} = Decoder.decode(data, decoder)
    assert chunk == StreamChunk.text("default")
  end

  test "custom protocol events can be skipped without changing provider state" do
    parser = fn _, state -> {:ok, [nil, %{data: "valid"}], state} end

    assert {:ok, [content, count], false, _} =
             Decoder.decode("ignored", new_decoder(protocol_parser: parser))

    assert content == StreamChunk.text("valid")
    assert count == StreamChunk.text("test:1")
  end

  defp new_decoder(opts \\ []) do
    Decoder.new(
      Keyword.merge(
        [provider_mod: StatefulProvider, model: %LLMDB.Model{provider: :test, id: "test"}],
        opts
      )
    )
  end
end
