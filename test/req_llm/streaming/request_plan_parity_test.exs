defmodule ReqLLM.Streaming.RequestPlanParityTest do
  use ExUnit.Case, async: true

  alias ReqLLM.Providers.{Anthropic, OpenAI}
  alias ReqLLM.RequestPlan

  @routes [
    {:openai, OpenAI, "openai_responses"},
    {:openai, OpenAI, "openai_chat"},
    {:anthropic, Anthropic, "anthropic_messages"},
    {:custom_stream, __MODULE__, nil}
  ]

  for {provider, provider_module, wire} <- @routes,
      transport <- [:finch, :websocket, :in_process] do
    @provider provider
    @provider_module provider_module
    @wire wire
    @transport transport

    test "deferred #{@provider} #{@wire} #{@transport} plans match accepted strict plans" do
      model = %LLMDB.Model{
        provider: @provider,
        id: "stream-model",
        extra: %{wire: %{protocol: @wire}}
      }

      opts = [temperature: 0.2, extension_schema: NimbleOptions.new!(value: [type: :string])]

      deferred =
        RequestPlan.from_resolved_stream(model, :chat, @provider_module, opts, @transport)

      strict = RequestPlan.from_resolved(model, :chat, @provider_module, opts, @transport)

      case strict do
        {:ok, plan} ->
          assert deferred == {:ok, plan}

        {:error, %ReqLLM.Error.Invalid.Parameter{}} ->
          assert {:ok, plan} = deferred
          assert plan.transport == :websocket
          assert plan.api_module in [OpenAI.ChatAPI, Anthropic]
          refute plan.surface == :openai_responses
      end
    end
  end

  for {provider, provider_module} <- [openai: OpenAI, anthropic: Anthropic] do
    @provider provider
    @provider_module provider_module

    test "invalid #{@provider} wire metadata stays owned by the provider callback" do
      model = %LLMDB.Model{
        provider: @provider,
        id: "stream-model",
        extra: %{wire: %{protocol: "unsupported"}}
      }

      assert {:error, %ReqLLM.Error.Invalid.Parameter{}} =
               RequestPlan.from_resolved(model, :chat, @provider_module, [], :finch)

      assert {:ok, plan} =
               RequestPlan.from_resolved_stream(model, :chat, @provider_module, [], :finch)

      assert plan.surface == :provider
      assert plan.api_module == @provider_module
      assert plan.provider_module == @provider_module
      assert plan.transport == :finch
    end
  end

  test "image and custom provider operations retain the provider-owned surface and options" do
    model = %LLMDB.Model{provider: :openai, id: "gpt-image-1.5"}
    opts = [operation: :image, stream: true]

    assert RequestPlan.from_resolved_stream(model, :image, OpenAI, opts, :finch) ==
             RequestPlan.from_resolved(model, :image, OpenAI, opts, :finch)

    assert {:ok, plan} =
             RequestPlan.from_resolved_stream(
               model,
               :custom_operation,
               __MODULE__,
               opts,
               :in_process
             )

    assert plan.operation == :custom_operation
    assert plan.surface == :provider
    assert plan.options == opts
  end
end
