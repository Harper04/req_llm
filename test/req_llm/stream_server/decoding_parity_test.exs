defmodule ReqLLM.StreamServer.DecodingParityTest do
  use ExUnit.Case, async: true

  import ExUnit.CaptureLog

  alias ReqLLM.{StreamChunk, StreamServer}
  alias ReqLLM.Providers.{AmazonBedrock, OpenAI}

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

  defp run_stream(provider, model, input, parser) do
    {:ok, server} =
      StreamServer.start_link(provider_mod: provider, model: model, protocol_parser: parser)

    on_exit(fn -> if Process.alive?(server), do: GenServer.stop(server) end)

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
