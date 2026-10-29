defmodule ReqLLM.Provider.AdapterDispatchTest do
  use ExUnit.Case, async: false

  alias ReqLLM.Provider.Adapter

  @moduletag contract: :provider_extension

  defmodule DefaultProvider do
    use ReqLLM.Provider,
      id: :adapter_dispatch_default,
      default_base_url: "https://adapter.example/v1"
  end

  defmodule BareProvider do
    @behaviour ReqLLM.Provider

    def provider_id, do: :adapter_dispatch_bare
    def prepare_request(_operation, _model, _input, _opts), do: {:error, :not_implemented}
    def attach(request, _model, _opts), do: request
    def encode_body(request), do: request
    def decode_response(request_response), do: request_response
  end

  setup_all do
    for provider <- [DefaultProvider, BareProvider] do
      assert {:ok, id} = ReqLLM.Providers.register(provider)
      on_exit(fn -> ReqLLM.Providers.unregister(id) end)
    end

    ReqLLM.Test.CatalogGatewayCatalog.install!()
  end

  test "registered default and bare providers take priority over catalog metadata" do
    for provider <- [DefaultProvider, BareProvider], operation <- [:chat, :object] do
      model = %LLMDB.Model{provider: provider.provider_id(), id: "not-in-the-catalog"}

      assert Adapter.resolve(model, operation, stream: true) == {:ok, provider}

      assert Adapter.resolve(model, operation, stream: true) ==
               ReqLLM.ProviderDispatch.get(model, operation, stream: true)
    end
  end

  test "all other operations use provider lookup without catalog fallback", %{model: model} do
    for operation <- [
          :embedding,
          :image,
          :speech,
          :transcription,
          :rerank,
          :ocr,
          :video,
          :video_query,
          :video_upload,
          :video_retrieve,
          :compact,
          :evaluate
        ] do
      for provider <- [DefaultProvider, BareProvider] do
        registered_model = %{model | provider: provider.provider_id()}
        assert Adapter.resolve(registered_model, operation) == {:ok, provider}

        assert Adapter.resolve(registered_model, operation) ==
                 ReqLLM.provider(registered_model.provider)
      end

      assert {:error, %ReqLLM.Error.Invalid.Provider{provider: :llmapi} = error} =
               Adapter.resolve(model, operation, stream: true)

      assert {:error, direct_error} = ReqLLM.provider(model.provider)
      assert %{error | stacktrace: direct_error.stacktrace} === direct_error
    end
  end

  test "chat and object retain the catalog gateway and stream capability checks", %{model: model} do
    object_contract = Map.fetch!(model.execution, :text)
    model = %{model | execution: Map.put(model.execution, :object, object_contract)}

    for operation <- [:chat, :object], opts <- [[], [stream: true]] do
      assert Adapter.resolve(model, operation, opts) == {:ok, ReqLLM.Providers.CatalogGateway}

      assert Adapter.resolve(model, operation, opts) ==
               ReqLLM.ProviderDispatch.get(model, operation, opts)
    end

    non_streaming_model = %{model | capabilities: %{chat: true}}

    assert {:error, %ReqLLM.Error.Invalid.Parameter{} = error} =
             Adapter.resolve(non_streaming_model, :chat, stream: true)

    assert {:error, direct_error} =
             ReqLLM.ProviderDispatch.get(non_streaming_model, :chat, stream: true)

    assert %{error | stacktrace: direct_error.stacktrace} === direct_error
    assert Exception.message(error) =~ "the model does not declare text streaming support"
  end

  test "catalog gateway rejection retains the operation and important error text", %{model: model} do
    assert {:error, %ReqLLM.Error.Invalid.Parameter{} = error} =
             Adapter.resolve(model, :object)

    assert {:error, direct_error} = ReqLLM.ProviderDispatch.get(model, :object)
    assert %{error | stacktrace: direct_error.stacktrace} === direct_error

    assert error.parameter ==
             "llmapi:vendor/chat-model cannot run object through the catalog gateway: " <>
               "the model needs a supported object execution contract with family " <>
               "openai_chat_compatible and wire protocol openai_chat"
  end
end
