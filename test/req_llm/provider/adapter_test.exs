defmodule ReqLLM.Provider.AdapterTest do
  use ExUnit.Case, async: true

  alias ReqLLM.Provider.Adapter

  @moduletag contract: :provider_extension

  defmodule DefaultProvider do
    use ReqLLM.Provider,
      id: :adapter_default,
      default_base_url: "https://adapter.example/v1"
  end

  defmodule ExplicitDefaultsProvider do
    use ReqLLM.Provider,
      id: :adapter_explicit_defaults,
      default_base_url: "https://adapter.example/v1"

    use ReqLLM.Provider.Defaults
  end

  defmodule BareProvider do
    @behaviour ReqLLM.Provider

    def prepare_request(operation, model, input, opts) do
      case Keyword.get(opts, :callback_action) do
        :raise -> raise "provider preparation failed"
        :throw -> throw(:provider_preparation_failed)
        :exit -> exit(:provider_preparation_failed)
        nil -> Keyword.get(opts, :callback_result, request(operation, model, input, opts))
      end
    end

    def attach(request, _model, _opts), do: request
    def encode_body(request), do: request
    def decode_response(request_response), do: request_response

    defp request(operation, model, input, opts) do
      {:ok, %Req.Request{private: %{callback_arguments: {operation, model, input, opts}}}}
    end
  end

  @operations [
    :chat,
    :object,
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
  ]

  test "buffered preparation preserves every operation and opaque callback arguments" do
    model = %{provider: :external_adapter, id: "not-in-the-catalog"}
    input = %{document: <<0, 1, 255>>, task: {"task-id", make_ref()}}
    schema = NimbleOptions.new!(name: [type: :string, required: true])

    opts = [
      temperature: 0.2,
      provider_options: [external: %{value: 1}],
      compiled_schema: schema,
      req_http_options: [retry: false],
      temperature: 0.9
    ]

    for operation <- @operations do
      assert {:ok, %Req.Request{private: %{callback_arguments: arguments}}} =
               Adapter.prepare_request(BareProvider, operation, model, input, opts)

      assert arguments === {operation, model, input, opts}

      assert Adapter.prepare_request(BareProvider, operation, model, input, opts) ===
               BareProvider.prepare_request(operation, model, input, opts)
    end
  end

  test "buffered preparation returns the provider's exact request or error value" do
    model = %LLMDB.Model{provider: :external_adapter, id: "external"}
    request = Req.Request.put_private(Req.new(), :opaque, make_ref())

    error =
      ReqLLM.Error.Invalid.Parameter.exception(parameter: "external provider rejected its input")

    for result <- [{:ok, request}, {:error, error}, {:error, {:external, make_ref()}}] do
      assert Adapter.prepare_request(BareProvider, :ocr, model, "document",
               callback_result: result
             ) ===
               result

      assert Adapter.prepare_request(BareProvider, :ocr, model, "document",
               callback_result: result
             ) ===
               BareProvider.prepare_request(:ocr, model, "document", callback_result: result)
    end
  end

  test "provider exceptions, throws and exits are not converted to new errors" do
    model = %LLMDB.Model{provider: :external_adapter, id: "external"}

    assert_raise RuntimeError, "provider preparation failed", fn ->
      Adapter.prepare_request(BareProvider, :ocr, model, "document", callback_action: :raise)
    end

    assert catch_throw(
             Adapter.prepare_request(BareProvider, :ocr, model, "document",
               callback_action: :throw
             )
           ) == :provider_preparation_failed

    assert catch_exit(
             Adapter.prepare_request(BareProvider, :ocr, model, "document",
               callback_action: :exit
             )
           ) == :provider_preparation_failed
  end

  for provider <- [DefaultProvider, ExplicitDefaultsProvider] do
    @provider provider

    test "#{inspect(provider)} keeps inherited buffered preparation and Req pipeline steps" do
      provider = @provider
      model = %LLMDB.Model{provider: provider.provider_id(), id: "external-model"}
      opts = [api_key: "adapter-test-key", temperature: 0.3]

      assert {:ok, %Req.Request{} = request} =
               Adapter.prepare_request(provider, :chat, model, "Hello", opts)

      assert {:ok, direct_request} = provider.prepare_request(:chat, model, "Hello", opts)
      direct_context = ReqLLM.Telemetry.request_context(direct_request)
      context = ReqLLM.Telemetry.request_context(request)
      refute context.request_id == direct_context.request_id

      assert ReqLLM.Telemetry.put_request_context(request, %{
               context
               | request_id: direct_context.request_id
             }) === direct_request

      assert request.url.path == "/chat/completions"
      assert request.options[:model] == "external-model"
      assert request.options[:temperature] == 0.3
      assert request.options[:auth] == {:bearer, "adapter-test-key"}
      assert request.request_steps[:llm_encode_body] == (&provider.encode_body/1)
      assert request.response_steps[:llm_decode_response] == (&provider.decode_response/1)
    end

    test "#{inspect(provider)} preserves its unsupported-operation error" do
      provider = @provider
      model = %LLMDB.Model{provider: provider.provider_id(), id: "external-model"}

      assert {:error, %ReqLLM.Error.Invalid.Parameter{} = error} =
               Adapter.prepare_request(provider, :ocr, model, "document", [])

      assert {:error, direct_error} = provider.prepare_request(:ocr, model, "document", [])
      assert %{error | stacktrace: direct_error.stacktrace} === direct_error
      assert Exception.message(error) =~ "operation: :ocr not supported by #{inspect(provider)}"
    end
  end
end
