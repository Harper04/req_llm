defmodule ReqLLM.RequestPlan.CompatibilityTest do
  use ExUnit.Case, async: false

  alias ReqLLM.Providers.{Anthropic, OpenAI}
  alias ReqLLM.RequestPlan

  for {provider, protocol, surface, provider_module, api_module, path} <- [
        {:openai, "openai_responses", :openai_responses, OpenAI, OpenAI.ResponsesAPI,
         "/responses"},
        {:openai, "openai_chat", :openai_chat_completions, OpenAI, OpenAI.ChatAPI,
         "/chat/completions"},
        {:anthropic, "anthropic_messages", :anthropic_messages, Anthropic, Anthropic,
         "/v1/messages"}
      ],
      operation <- [:chat, :object],
      {stream, transport} <- [{false, :req}, {true, :finch}] do
    @provider provider
    @protocol protocol
    @surface surface
    @provider_module provider_module
    @api_module api_module
    @path path
    @operation operation
    @stream stream
    @transport transport

    test "pins #{@surface} #{@operation} #{@transport} plans and diagnostics" do
      model =
        ReqLLM.model!(%{
          provider: @provider,
          id: "custom-model",
          extra: %{wire: %{protocol: @protocol}}
        })

      assert RequestPlan.build(model, @operation, stream: @stream) ==
               {:ok,
                %RequestPlan{
                  model: model,
                  operation: @operation,
                  provider: @provider,
                  surface: @surface,
                  transport: @transport,
                  options: [stream: @stream],
                  provider_module: @provider_module,
                  api_module: @api_module,
                  warnings: []
                }}

      assert ReqLLM.plan(model, @operation, stream: @stream) ==
               {:ok,
                %{
                  model: %{provider: @provider, id: "custom-model"},
                  operation: @operation,
                  surface: @surface,
                  transport: @transport,
                  route: %{method: :post, path: @path},
                  options: %{canonical: [:stream], translated: [:stream]},
                  fallbacks: [],
                  warnings: []
                }}
    end
  end

  test "pins the unsupported-operation errors at both planning boundaries" do
    assert {:error, %ReqLLM.Error.Invalid.Parameter{parameter: parameter}} =
             RequestPlan.build("openai:gpt-4o-mini", :image)

    assert parameter == "request planning supports only :chat and :object operations, got: :image"

    assert {:error, %ReqLLM.Error.Invalid.Parameter{parameter: parameter}} =
             ReqLLM.plan("openai:gpt-4o-mini", :image)

    assert parameter == "request planning supports only :chat and :object operations"
  end
end
