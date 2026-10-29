defmodule ReqLLM.StreamServer.DecodingParityTest do
  use ExUnit.Case, async: true

  import ExUnit.CaptureLog

  alias ReqLLM.Providers.{AmazonBedrock, Groq, OpenAI}
  alias ReqLLM.{StreamChunk, StreamServer, Telemetry}

  defmodule FlushProvider do
    @moduledoc false

    def init_stream_state(_model), do: 0

    def decode_stream_event(%{data: "pending"}, _model, count) do
      {[StreamChunk.thinking("pending")], count + 1}
    end

    def flush_stream_state(model, count) do
      send(model.extra.test_pid, {:flush_order, {:provider_flush, count}})
      {[StreamChunk.text("suffix:#{count}")], count}
    end
  end

  test "SSE preserves content, metadata and order across byte boundaries" do
    model = %LLMDB.Model{
      provider: :openai,
      id: "gpt-4o",
      extra: %{wire: %{protocol: "openai_chat"}}
    }

    input =
      Enum.map_join(
        [
          %{"choices" => [%{"index" => 0, "delta" => %{"content" => "Hello"}}]},
          %{"choices" => [%{"index" => 0, "delta" => %{"content" => " world"}}]},
          %{"choices" => [%{"index" => 0, "delta" => %{}, "finish_reason" => "stop"}]}
        ],
        fn data -> "data: #{Jason.encode!(data)}\n\n" end
      ) <> "data: [DONE]\n\n"

    chunks = run_stream(OpenAI, model, input, nil)

    assert chunks == [
             StreamChunk.text("Hello"),
             StreamChunk.text(" world"),
             StreamChunk.meta(%{finish_reason: :stop}),
             StreamChunk.meta(%{terminal?: true})
           ]
  end

  test "Bedrock binary event-stream preserves partial frames and decoded content order" do
    model = %LLMDB.Model{provider: :amazon_bedrock, id: "openai.gpt-5.6-terra"}

    input =
      Enum.map_join(["Hello", " world"], fn text ->
        delta = %{
          "object" => "chat.completion.chunk",
          "choices" => [%{"index" => 0, "delta" => %{"content" => text}}]
        }

        payload = Jason.encode!(%{"bytes" => Base.encode64(Jason.encode!(delta))})
        frame(payload)
      end)

    chunks =
      run_stream(AmazonBedrock, model, input, AmazonBedrock.stream_protocol_parser(model, []))

    assert chunks == [StreamChunk.text("Hello"), StreamChunk.text(" world")]
  end

  test "parse errors warn and retain the previous protocol buffer" do
    parser = fn
      "partial", nil ->
        {:incomplete, "saved"}

      "bad", "saved" ->
        {:error, :invalid_frame}

      "rest", "saved" ->
        {:ok, [%{data: %{"choices" => [%{"delta" => %{"content" => "recovered"}}]}}], ""}
    end

    {:ok, server} =
      StreamServer.start_link(
        provider_mod: OpenAI,
        model: %LLMDB.Model{
          provider: :openai,
          id: "gpt-4o",
          extra: %{wire: %{protocol: "openai_chat"}}
        },
        protocol_parser: parser
      )

    assert :ok = StreamServer.http_event(server, {:data, "partial"})

    assert capture_log(fn ->
             assert :ok = StreamServer.http_event(server, {:data, "bad"})
           end) =~ "Protocol parse error: :invalid_frame"

    assert :ok = StreamServer.http_event(server, {:data, "rest"})
    assert :ok = StreamServer.http_event(server, :done)
    assert collect(server) == [StreamChunk.text("recovered")]
  end

  test "HTTP fallback discards a custom parser and pending bytes" do
    parser = fn _, _ -> {:incomplete, :custom_protocol_state} end

    {:ok, server} =
      StreamServer.start_link(
        provider_mod: OpenAI,
        model: %LLMDB.Model{
          provider: :openai,
          id: "gpt-4o",
          extra: %{wire: %{protocol: "openai_chat"}}
        },
        protocol_parser: parser
      )

    assert :ok = StreamServer.http_event(server, {:data, "pending"})
    assert :ok = StreamServer.http_event(server, {:transport_fallback, :http, :test, nil, nil})

    assert :ok =
             StreamServer.http_event(
               server,
               {:data, ~s(data: {"choices":[{"delta":{"content":"http"}}]}\n\n)}
             )

    assert :ok = StreamServer.http_event(server, :done)
    assert collect(server) == [StreamChunk.text("http")]
  end

  test "unterminated Groq SSE emits protocol content before the buffered provider suffix" do
    model = %LLMDB.Model{provider: :groq, id: "qwen/qwen3-32b"}
    server = start_supervised!({StreamServer, provider_mod: Groq, model: model})
    input = ~s(data: {"choices":[{"delta":{"content":"Hello world!"}}]})

    assert :ok = StreamServer.http_event(server, {:data, input})
    assert :ok = StreamServer.http_event(server, :done)
    assert collect(server) == [StreamChunk.text("Hello "), StreamChunk.text("world!")]
  end

  test "protocol flush emits chunk telemetry before invoking the provider flush callback" do
    model = %LLMDB.Model{
      provider: :anthropic,
      id: "test-reasoning-model",
      capabilities: %{reasoning: %{enabled: true}},
      extra: %{test_pid: self()}
    }

    context =
      model
      |> Telemetry.new_context([reasoning_effort: :high], mode: :stream, transport: :finch)
      |> Telemetry.start_request(%{
        "thinking" => %{"type" => "enabled", "budget_tokens" => 4096}
      })

    handler = "flush-order-#{System.unique_integer([:positive])}"

    :ok =
      :telemetry.attach(
        handler,
        [:req_llm, :reasoning, :update],
        &__MODULE__.observe_flush/4,
        {self(), context.request_id}
      )

    on_exit(fn -> :telemetry.detach(handler) end)
    server = start_supervised!({StreamServer, provider_mod: FlushProvider, model: model})

    assert :ok = StreamServer.set_telemetry_context(server, context)
    assert :ok = StreamServer.http_event(server, {:data, "data: pending"})
    assert :ok = StreamServer.http_event(server, :done)

    assert_receive {:flush_order, first}
    assert_receive {:flush_order, second}
    assert [first, second] == [:protocol_chunk_telemetry, {:provider_flush, 1}]
    assert collect(server) == [StreamChunk.thinking("pending"), StreamChunk.text("suffix:1")]
  end

  def observe_flush(
        _event,
        _measurements,
        %{request_id: id, milestone: :content_started},
        {pid, id}
      ) do
    send(pid, {:flush_order, :protocol_chunk_telemetry})
  end

  def observe_flush(_event, _measurements, _metadata, _config), do: :ok

  defp run_stream(provider, model, input, parser) do
    server =
      start_supervised!(
        {StreamServer, provider_mod: provider, model: model, protocol_parser: parser}
      )

    for <<byte <- input>> do
      assert :ok = StreamServer.http_event(server, {:data, <<byte>>})
    end

    assert :ok = StreamServer.http_event(server, :done)
    collect(server)
  end

  defp collect(server) do
    case StreamServer.next(server, 1_000) do
      {:ok, chunk} -> [chunk | collect(server)]
      :halt -> []
    end
  end

  defp frame(payload) do
    prelude = <<byte_size(payload) + 16::32-big, 0::32-big>>
    message = <<prelude::binary, :erlang.crc32(prelude)::32-big, payload::binary>>
    <<message::binary, :erlang.crc32(message)::32-big>>
  end
end
