defmodule ReqLLM.Streaming.StartupParityTest do
  use ExUnit.Case, async: true

  alias ReqLLM.{Context, StreamChunk, Streaming, StreamResponse}
  alias ReqLLM.Providers.{Anthropic, OpenAI}

  defmodule MissingHttpProvider do
  end

  defmodule MissingWebSocketProvider do
    def stream_transport(_model, _opts), do: :websocket
  end

  defmodule MissingInProcessProvider do
    def stream_transport(_model, _opts), do: :in_process
  end

  defmodule EventStreamProvider do
    def stream_protocol_parser(_model, opts) do
      send(opts[:owner], :selected_event_stream_parser)
      &ReqLLM.Providers.AmazonBedrock.parse_stream_protocol/2
    end

    def attach_stream(_model, _context, opts, finch_name) do
      send(opts[:owner], {:attached_http, finch_name, opts})
      {:ok, Finch.build(:post, opts[:base_url], [], "{}")}
    end

    def decode_stream_event(%{"text" => text}, _model), do: [StreamChunk.text(text)]
    def decode_stream_event(%{"done" => true}, _model), do: [StreamChunk.meta(%{terminal?: true})]
  end

  defmodule Endpoint do
    def init(opts), do: opts

    def call(conn, opts) do
      conn
      |> Plug.Conn.put_resp_content_type("application/vnd.amazon.eventstream")
      |> Plug.Conn.send_resp(200, opts[:body])
    end
  end

  test "OpenAI Chat WebSocket rejection retains the provider's request error and text" do
    model = %LLMDB.Model{
      provider: :openai,
      id: "gpt-4o",
      extra: %{wire: %{protocol: "openai_chat"}}
    }

    assert {:error, {:websocket_streaming_failed, {:provider_build_failed, error}}} =
             Streaming.start_stream(OpenAI, model, Context.new([Context.user("Hello")]),
               provider_options: [openai_stream_transport: :websocket]
             )

    assert %ReqLLM.Error.API.Request{} = error

    assert error.reason ==
             "OpenAI WebSocket mode is only supported for Responses models. openai:gpt-4o routes to ReqLLM.Providers.OpenAI.ChatAPI."

    assert Exception.message(error) == "API request failed: #{error.reason}"
  end

  for {provider_id, provider} <- [openai: OpenAI, anthropic: Anthropic] do
    @provider_id provider_id
    @provider provider

    test "#{provider_id} invalid wire metadata retains its parameter error and text" do
      model = %LLMDB.Model{
        provider: @provider_id,
        id: "invalid-wire",
        extra: %{wire: %{protocol: "unsupported"}}
      }

      assert {:error, {:http_streaming_failed, {:provider_build_failed, error}}} =
               Streaming.start_stream(@provider, model, Context.new([Context.user("Hello")]))

      assert %ReqLLM.Error.Invalid.Parameter{} = error

      assert error.parameter ==
               "wire protocol \"unsupported\" is invalid for provider #{@provider_id |> inspect()}"

      assert Exception.message(error) == "Invalid parameter: #{error.parameter}"
    end
  end

  test "Anthropic WebSocket options retain the existing HTTP provider rejection" do
    model = %LLMDB.Model{provider: :anthropic, id: "claude-test"}

    assert {:error, {:http_streaming_failed, {:provider_build_failed, error}}} =
             Streaming.start_stream(Anthropic, model, Context.new([Context.user("Hello")]),
               provider_options: [openai_stream_transport: :websocket]
             )

    assert %ReqLLM.Error.Invalid.Parameter{} = error
    assert error.parameter == "WebSocket transport is not supported by Anthropic Messages"
    assert Exception.message(error) == "Invalid parameter: #{error.parameter}"
  end

  test "missing HTTP callbacks retain the UndefinedFunctionError wrapper" do
    assert {:error, {:http_streaming_failed, {:build_request_failed, error}}} =
             Streaming.start_stream(
               MissingHttpProvider,
               model(),
               Context.new([Context.user("Hello")])
             )

    assert %UndefinedFunctionError{
             module: MissingHttpProvider,
             function: :attach_stream,
             arity: 4
           } = error
  end

  test "missing WebSocket callbacks retain the model-specific request error" do
    assert {:error, {:websocket_streaming_failed, error}} =
             Streaming.start_stream(
               MissingWebSocketProvider,
               model(),
               Context.new([Context.user("Hello")])
             )

    assert %ReqLLM.Error.API.Request{} = error

    assert error.reason ==
             "ReqLLM.Streaming.StartupParityTest.MissingWebSocketProvider does not implement WebSocket streaming for startup_parity:model"
  end

  test "missing in-process callbacks retain the missing_callback tuple" do
    assert {:error,
            {:in_process_streaming_failed,
             {:provider_build_failed, {:missing_callback, :attach_in_process_stream, 3}}}} =
             Streaming.start_stream(
               MissingInProcessProvider,
               model(),
               Context.new([Context.user("Hello")])
             )
  end

  test "HTTP startup keeps the selected Bedrock event-stream parser, pool and extension options" do
    payload =
      frame(%{"text" => "first"}) <> frame(%{"text" => "second"}) <> frame(%{"done" => true})

    server =
      start_supervised!({Bandit, plug: {Endpoint, body: payload}, ip: {127, 0, 0, 1}, port: 0})

    {:ok, {_address, port}} = ThousandIsland.listener_info(server)
    schema = NimbleOptions.new!(value: [type: :string])

    assert {:ok, response} =
             Streaming.start_stream(
               EventStreamProvider,
               model(),
               Context.new([Context.user("Hello")]),
               owner: self(),
               base_url: "http://127.0.0.1:#{port}/events",
               finch_name: ReqLLM.Finch,
               extension_schema: schema,
               pricing_context: %{pricing_period: "peak"}
             )

    assert_receive :selected_event_stream_parser
    assert_receive {:attached_http, ReqLLM.Finch, opts}
    assert opts[:extension_schema] === schema
    refute Keyword.has_key?(opts, :pricing_context)
    refute Keyword.has_key?(opts, :defer_http_events_until_telemetry?)
    assert StreamResponse.text(response) == "firstsecond"
    assert StreamResponse.finish_reason(response) == :stop
  end

  defp model, do: %LLMDB.Model{provider: :startup_parity, id: "model"}

  defp frame(event) do
    payload = Jason.encode!(event)
    prelude = <<byte_size(payload) + 16::32-big, 0::32-big>>
    message = prelude <> <<:erlang.crc32(prelude)::32-big>> <> payload
    message <> <<:erlang.crc32(message)::32-big>>
  end
end
