defmodule ReqLLM.RequestPlan.ResolvedTest do
  use ExUnit.Case, async: true

  alias ReqLLM.Providers
  alias ReqLLM.RequestPlan

  for {operation, provider, provider_module} <- [
        {:chat, :google, Providers.Google},
        {:object, :amazon_bedrock, Providers.AmazonBedrock},
        {:embedding, :openai, Providers.OpenAI},
        {:image, :openai, Providers.OpenAI},
        {:speech, :elevenlabs, Providers.ElevenLabs},
        {:transcription, :openai, Providers.OpenAI},
        {:rerank, :cohere, Providers.Cohere},
        {:ocr, :google_vertex, Providers.GoogleVertex},
        {:video, :minimax, Providers.Minimax},
        {:video_query, :minimax, Providers.Minimax},
        {:video_upload, :minimax, Providers.Minimax},
        {:video_retrieve, :minimax, Providers.Minimax},
        {:compact, :openai, Providers.OpenAI},
        {:evaluate, :openai, Providers.OpenAI}
      ] do
    @operation operation
    @provider provider
    @provider_module provider_module

    test "represents the resolved #{@provider} #{@operation} callback path" do
      model = LLMDB.Model.new!(%{provider: @provider, id: "custom-model"})
      opts = [receive_timeout: 1234, provider_options: [custom_option: "unchanged"]]

      assert RequestPlan.from_resolved(model, @operation, @provider_module, opts, :req) ==
               {:ok,
                %RequestPlan{
                  model: model,
                  operation: @operation,
                  provider: @provider,
                  provider_module: @provider_module,
                  api_module: @provider_module,
                  surface: :provider,
                  transport: :req,
                  options: opts,
                  warnings: []
                }}
    end
  end

  test "reuses the existing OpenAI and Anthropic surface decisions" do
    for {provider, model_id, provider_module} <- [
          {:openai, "gpt-4o-mini", Providers.OpenAI},
          {:openai, "chat-latest", Providers.OpenAI},
          {:anthropic, "claude-sonnet-4-5-20250929", Providers.Anthropic}
        ],
        operation <- [:chat, :object],
        {opts, transport} <- [
          {[stream: false], :req},
          {[stream: true], :finch}
        ] do
      model = ReqLLM.model!(%{provider: provider, id: model_id})

      assert RequestPlan.from_resolved(model, operation, provider_module, opts, transport) ==
               RequestPlan.build(model, operation, opts)
    end
  end

  test "retains boundary options exactly without merging, sorting, or translating them" do
    model = LLMDB.Model.new!(%{provider: :google, id: "custom-model"})

    opts = [
      temperature: 0.3,
      provider_options: %{google: [custom_option: "value"]},
      temperature: 0.7,
      stream?: true
    ]

    assert {:ok, plan} =
             RequestPlan.from_resolved(model, :chat, Providers.Google, opts, :finch)

    assert plan.options == opts
    assert plan.model == model
    assert plan.transport == :finch
  end

  test "represents image streaming independently of the model's chat wire metadata" do
    model =
      LLMDB.Model.new!(%{
        provider: :openai,
        id: "gpt-image-1",
        extra: %{wire: %{protocol: "openai_responses"}}
      })

    assert {:ok, plan} =
             RequestPlan.from_resolved(model, :image, Providers.OpenAI, [stream: true], :finch)

    assert plan.surface == :provider
    assert plan.api_module == Providers.OpenAI
    assert plan.transport == :finch
  end

  test "retains a selected custom provider and in-process transport without looking them up" do
    model = LLMDB.Model.new!(%{provider: :custom_provider, id: "private-model"})

    assert {:ok, plan} =
             RequestPlan.from_resolved(model, :chat, CustomProvider, [], :in_process)

    assert plan.provider == :custom_provider
    assert plan.provider_module == CustomProvider
    assert plan.api_module == CustomProvider
    assert plan.surface == :provider
    assert plan.transport == :in_process
  end

  test "retains the CatalogGateway module selected by the generation boundary" do
    model = LLMDB.Model.new!(%{provider: :perplexity, id: "private-model"})

    for operation <- [:chat, :object] do
      assert {:ok, plan} =
               RequestPlan.from_resolved(model, operation, Providers.CatalogGateway, [], :req)

      assert plan.provider == :perplexity
      assert plan.provider_module == Providers.CatalogGateway
      assert plan.surface == :provider
    end
  end

  test "retains the selected WebSocket transport for a provider-owned route" do
    model = LLMDB.Model.new!(%{provider: :openai_codex, id: "private-model"})

    assert {:ok, plan} =
             RequestPlan.from_resolved(model, :chat, Providers.OpenAICodex, [], :websocket)

    assert plan.transport == :websocket
    assert plan.surface == :provider
  end

  test "rejects unsupported operation and transport descriptors" do
    model = LLMDB.Model.new!(%{provider: :openai, id: "custom-model"})

    assert {:error, %ReqLLM.Error.Invalid.Parameter{parameter: parameter}} =
             RequestPlan.from_resolved(model, :unknown, Providers.OpenAI, [], :req)

    assert parameter == "unsupported request-plan operation: :unknown"

    assert {:error, %ReqLLM.Error.Invalid.Parameter{parameter: parameter}} =
             RequestPlan.from_resolved(model, :image, Providers.OpenAI, [], :udp)

    assert parameter == "unsupported request-plan transport: :udp"
  end

  test "rejects non-keyword options" do
    model = LLMDB.Model.new!(%{provider: :openai, id: "custom-model"})

    for opts <- [%{}, [:invalid], [{"key", true}]] do
      assert {:error, %ReqLLM.Error.Invalid.Parameter{}} =
               RequestPlan.from_resolved(model, :image, Providers.OpenAI, opts, :req)
    end
  end
end
