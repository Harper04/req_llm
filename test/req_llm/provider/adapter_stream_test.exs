defmodule ReqLLM.Provider.AdapterStreamTest do
  use ExUnit.Case, async: true

  alias ReqLLM.Provider.Adapter

  @moduletag contract: :provider_extension

  defmodule MinimalProvider do
    @behaviour ReqLLM.Provider

    def prepare_request(_operation, _model, _input, _opts), do: {:error, :not_implemented}
    def attach(request, _model, _opts), do: request
    def encode_body(request), do: request
    def decode_response(request_response), do: request_response
  end

  defmodule CustomProvider do
    @behaviour ReqLLM.Provider

    def prepare_request(_operation, _model, _input, _opts), do: {:error, :not_implemented}
    def attach(request, _model, _opts), do: request
    def encode_body(request), do: request
    def decode_response(request_response), do: request_response
    def stream_transport(_model, opts), do: Keyword.fetch!(opts, :transport)
    def stream_protocol_parser(_model, opts), do: Keyword.fetch!(opts, :parser)

    def attach_stream(model, context, opts, finch_name) do
      request =
        Finch.build(:post, "https://adapter.example/custom")
        |> Finch.Request.put_private(:callback_arguments, {model, context, opts, finch_name})

      result(opts, {:ok, request})
    end

    def attach_in_process_stream(model, context, opts) do
      result(
        opts,
        {:ok, [ReqLLM.StreamChunk.meta(%{callback_arguments: {model, context, opts}})]}
      )
    end

    def attach_websocket_stream(model, context, opts) do
      result(
        opts,
        {:ok, %{url: "wss://adapter.example", callback_arguments: {model, context, opts}}}
      )
    end

    defp result(opts, default) do
      case Keyword.get(opts, :callback_action) do
        :raise -> raise "stream startup failed"
        :throw -> throw(:stream_startup_failed)
        :exit -> exit(:stream_startup_failed)
        nil -> Keyword.get(opts, :callback_result, default)
      end
    end
  end

  defmodule DefaultProvider do
    use ReqLLM.Provider,
      id: :adapter_stream_default,
      default_base_url: "https://adapter.example/v1"
  end

  defmodule ExplicitDefaultsProvider do
    use ReqLLM.Provider,
      id: :adapter_stream_explicit,
      default_base_url: "https://adapter.example/v1"

    use ReqLLM.Provider.Defaults
  end

  defmodule LegacyProvider do
    use ReqLLM.Provider,
      id: :adapter_stream_legacy,
      default_base_url: "https://adapter.example/v1"

    def streaming_http(model, api_key, _opts) do
      %{path: "/legacy/#{model.id}", headers: [{"x-legacy-key", api_key}]}
    end
  end

  setup do
    {:ok, context} = ReqLLM.Context.normalize("Hello")
    %{model: %LLMDB.Model{provider: :external_adapter, id: "custom-model"}, context: context}
  end

  test "optional stream selectors retain the HTTP and nil fallbacks", %{model: model} do
    assert Adapter.stream_transport(MinimalProvider, model, []) == :http
    assert Adapter.stream_protocol_parser(MinimalProvider, model, []) == nil
  end

  test "custom selectors return every transport and the exact parser", %{model: model} do
    for transport <- [:http, :websocket, :in_process] do
      assert Adapter.stream_transport(CustomProvider, model, transport: transport) === transport

      assert Adapter.stream_transport(CustomProvider, model, transport: transport) ===
               CustomProvider.stream_transport(model, transport: transport)
    end

    parser = &ReqLLM.Streaming.SSE.accumulate_and_parse/2
    assert Adapter.stream_protocol_parser(CustomProvider, model, parser: parser) === parser

    assert Adapter.stream_protocol_parser(CustomProvider, model, parser: parser) ===
             CustomProvider.stream_protocol_parser(model, parser: parser)

    assert Adapter.stream_protocol_parser(CustomProvider, model, parser: nil) == nil
  end

  test "Finch startup preserves all arguments and its request", %{model: model, context: context} do
    opts = [provider_options: [custom: make_ref()], temperature: 0.2, temperature: 0.9]

    assert {:ok, %Finch.Request{private: %{callback_arguments: arguments}}} =
             Adapter.attach_stream(CustomProvider, model, context, opts, CustomFinch)

    assert arguments === {model, context, opts, CustomFinch}

    assert Adapter.attach_stream(CustomProvider, model, context, opts, CustomFinch) ===
             CustomProvider.attach_stream(model, context, opts, CustomFinch)
  end

  test "in-process startup preserves canonical chunks and cancellation without running them", %{
    model: model,
    context: context
  } do
    opts = [custom: make_ref()]

    assert {:ok, [%ReqLLM.StreamChunk{metadata: %{callback_arguments: arguments}}]} =
             Adapter.attach_in_process_stream(CustomProvider, model, context, opts)

    assert arguments === {model, context, opts}

    assert Adapter.attach_in_process_stream(CustomProvider, model, context, opts) ===
             CustomProvider.attach_in_process_stream(model, context, opts)

    stream = Stream.map([:not_enumerated], fn _ -> flunk("startup must not enumerate") end)
    cancel = fn -> flunk("startup must not cancel") end
    result = {:ok, ReqLLM.Provider.InProcessStream.new(stream, cancel: cancel)}

    assert Adapter.attach_in_process_stream(CustomProvider, model, context,
             callback_result: result
           ) ===
             result

    assert Adapter.attach_in_process_stream(CustomProvider, model, context,
             callback_result: result
           ) ===
             CustomProvider.attach_in_process_stream(model, context, callback_result: result)
  end

  test "WebSocket startup preserves the complete provider-owned configuration", %{
    model: model,
    context: context
  } do
    opts = [custom: make_ref(), req_http_options: [headers: [{"x-test", "value"}]]]

    assert {:ok, %{url: "wss://adapter.example", callback_arguments: arguments}} =
             Adapter.attach_websocket_stream(CustomProvider, model, context, opts)

    assert arguments === {model, context, opts}

    assert Adapter.attach_websocket_stream(CustomProvider, model, context, opts) ===
             CustomProvider.attach_websocket_stream(model, context, opts)
  end

  test "missing optional startup callbacks retain distinct detection behavior", %{
    model: model,
    context: context
  } do
    assert Adapter.attach_in_process_stream(MinimalProvider, model, context, []) ==
             {:error, {:missing_callback, :attach_in_process_stream, 3}}

    assert Adapter.attach_websocket_stream(MinimalProvider, model, context, []) == false

    error =
      assert_raise UndefinedFunctionError, fn ->
        Adapter.attach_stream(MinimalProvider, model, context, [], CustomFinch)
      end

    assert {error.module, error.function, error.arity} == {MinimalProvider, :attach_stream, 4}
  end

  test "startup errors and non-local returns stay provider-owned", %{
    model: model,
    context: context
  } do
    callbacks = [
      fn opts -> Adapter.attach_stream(CustomProvider, model, context, opts, CustomFinch) end,
      fn opts -> Adapter.attach_in_process_stream(CustomProvider, model, context, opts) end,
      fn opts -> Adapter.attach_websocket_stream(CustomProvider, model, context, opts) end
    ]

    for callback <- callbacks do
      error = {:error, {:provider_rejected, make_ref()}}
      assert callback.(callback_result: error) === error

      assert_raise RuntimeError, "stream startup failed", fn ->
        callback.(callback_action: :raise)
      end

      assert catch_throw(callback.(callback_action: :throw)) == :stream_startup_failed
      assert catch_exit(callback.(callback_action: :exit)) == :stream_startup_failed
    end
  end

  for {provider, path, auth_header} <- [
        {DefaultProvider, "/v1/chat/completions", {"Authorization", "Bearer adapter-test-key"}},
        {ExplicitDefaultsProvider, "/v1/chat/completions",
         {"Authorization", "Bearer adapter-test-key"}},
        {LegacyProvider, "/v1/legacy/custom-model", {"x-legacy-key", "adapter-test-key"}}
      ] do
    @provider provider
    @path path
    @auth_header auth_header

    test "#{inspect(provider)} keeps inherited Finch preparation and legacy HTTP configuration",
         %{
           context: context
         } do
      provider = @provider
      model = %LLMDB.Model{provider: provider.provider_id(), id: "custom-model"}
      opts = [api_key: "adapter-test-key", stream: true, temperature: 0.3]

      assert {:ok, %Finch.Request{} = request} =
               Adapter.attach_stream(provider, model, context, opts, CustomFinch)

      assert {:ok, request} === provider.attach_stream(model, context, opts, CustomFinch)

      body = request.body |> IO.iodata_to_binary() |> Jason.decode!()
      assert body["model"] == "custom-model"
      assert body["stream"] == true
      assert body["temperature"] == 0.3

      assert request.path == @path
      assert @auth_header in request.headers
    end
  end
end
