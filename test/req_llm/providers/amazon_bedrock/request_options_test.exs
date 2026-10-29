defmodule ReqLLM.Providers.AmazonBedrock.RequestOptionsTest do
  use ExUnit.Case, async: true

  import ReqLLM.Test.AmazonBedrockHelpers, only: [bedrock_opts: 1, sparse_adaptive_opus: 0]

  alias ReqLLM.{Context, Providers.AmazonBedrock}

  describe "adaptive thinking for hosted Claude models" do
    test "encodes adaptive thinking when sparse Opus platform metadata requests reasoning" do
      model = %LLMDB.Model{
        id: "anthropic.claude-opus-4-7-v1:0",
        model: "anthropic.claude-opus-4-7-v1:0",
        provider: :amazon_bedrock,
        provider_model_id: "us.anthropic.claude-opus-4-7-v1:0",
        capabilities: %{chat: true, reasoning: %{enabled: true}},
        extra: %{family: "claude-opus"}
      }

      context = Context.new([Context.user("Hello")])

      opts = [
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST",
        region: "us-east-1",
        max_tokens: 2000,
        reasoning_effort: :high
      ]

      {:ok, request} = AmazonBedrock.prepare_request(:chat, model, context, opts)

      body = ReqLLM.Test.Helpers.json_body(request)
      assert body["thinking"] == %{"display" => "summarized", "type" => "adaptive"}
      assert request.url.path == "/model/us.anthropic.claude-opus-4-7-v1:0/invoke"
    end

    test "does not auto-enable adaptive thinking without reasoning params" do
      model = %LLMDB.Model{
        id: "anthropic.claude-opus-4-7-v1:0",
        model: "anthropic.claude-opus-4-7-v1:0",
        provider: :amazon_bedrock,
        provider_model_id: "us.anthropic.claude-opus-4-7-v1:0",
        capabilities: %{chat: true, reasoning: %{enabled: true}},
        extra: %{family: "claude-opus"}
      }

      context = Context.new([Context.user("Hello")])

      opts = [
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST",
        region: "us-east-1",
        max_tokens: 2000
      ]

      {:ok, request} = AmazonBedrock.prepare_request(:chat, model, context, opts)

      body = ReqLLM.Test.Helpers.json_body(request)
      refute Map.has_key?(body, "thinking")
    end

    test "sends the requested effort as output_config.effort for adaptive-only models" do
      for {effort, wire} <- [low: "low", medium: "medium", high: "high"] do
        {:ok, request} =
          AmazonBedrock.prepare_request(
            :chat,
            sparse_adaptive_opus(),
            Context.new([Context.user("Hello")]),
            bedrock_opts(reasoning_effort: effort)
          )

        body = ReqLLM.Test.Helpers.json_body(request)
        assert body["thinking"] == %{"display" => "summarized", "type" => "adaptive"}
        assert body["output_config"] == %{"effort" => wire}
      end
    end

    test "sends output_config.effort on the bedrock-mantle Messages route too" do
      {:ok, request} =
        AmazonBedrock.prepare_request(
          :chat,
          sparse_adaptive_opus(),
          Context.new([Context.user("Hello")]),
          bedrock_opts(reasoning_effort: :low, provider_options: [endpoint: :mantle])
        )

      body = ReqLLM.Test.Helpers.json_body(request)
      assert body["output_config"] == %{"effort" => "low"}
    end

    test "budget-thinking models get budget_tokens and no output_config" do
      model = %LLMDB.Model{
        id: "anthropic.claude-3-7-sonnet-20250219-v1:0",
        model: "anthropic.claude-3-7-sonnet-20250219-v1:0",
        provider: :amazon_bedrock,
        provider_model_id: "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        capabilities: %{chat: true, reasoning: %{enabled: true}},
        extra: %{family: "claude-sonnet"}
      }

      {:ok, request} =
        AmazonBedrock.prepare_request(
          :chat,
          model,
          Context.new([Context.user("Hello")]),
          bedrock_opts(reasoning_effort: :low, max_tokens: 8000)
        )

      body = ReqLLM.Test.Helpers.json_body(request)
      assert %{"type" => "enabled", "budget_tokens" => _} = body["thinking"]
      refute Map.has_key?(body, "output_config")
    end

    test "reasoning_effort :none sends thinking disabled and no output_config" do
      {:ok, request} =
        AmazonBedrock.prepare_request(
          :chat,
          sparse_adaptive_opus(),
          Context.new([Context.user("Hello")]),
          bedrock_opts(reasoning_effort: :none)
        )

      body = ReqLLM.Test.Helpers.json_body(request)
      assert body["thinking"] == %{"type" => "disabled"}
      refute Map.has_key?(body, "output_config")
    end
  end

  describe "service_tier parameter" do
    test "includes service_tier in request body when specified" do
      System.put_env("AWS_ACCESS_KEY_ID", "AKIATEST")
      System.put_env("AWS_SECRET_ACCESS_KEY", "secretTEST")

      {:ok, model} = ReqLLM.model("amazon-bedrock:anthropic.claude-3-haiku-20240307-v1:0")

      messages = [Context.user("Hello")]
      context = Context.new(messages)

      opts = [
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST",
        service_tier: "priority"
      ]

      {:ok, request} = AmazonBedrock.prepare_request(:chat, model, context, opts)

      assert %Req.Request{} = request
      body = ReqLLM.Test.Helpers.json_body(request)
      assert body["service_tier"] == "priority"
    end

    test "includes service_tier=flex in request body" do
      System.put_env("AWS_ACCESS_KEY_ID", "AKIATEST")
      System.put_env("AWS_SECRET_ACCESS_KEY", "secretTEST")

      {:ok, model} = ReqLLM.model("amazon-bedrock:anthropic.claude-3-haiku-20240307-v1:0")

      messages = [Context.user("Hello")]
      context = Context.new(messages)

      opts = [
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST",
        service_tier: "flex"
      ]

      {:ok, request} = AmazonBedrock.prepare_request(:chat, model, context, opts)

      body = ReqLLM.Test.Helpers.json_body(request)
      assert body["service_tier"] == "flex"
    end

    test "omits service_tier when default" do
      System.put_env("AWS_ACCESS_KEY_ID", "AKIATEST")
      System.put_env("AWS_SECRET_ACCESS_KEY", "secretTEST")

      {:ok, model} = ReqLLM.model("amazon-bedrock:anthropic.claude-3-haiku-20240307-v1:0")

      messages = [Context.user("Hello")]
      context = Context.new(messages)

      opts = [
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST",
        service_tier: "default"
      ]

      {:ok, request} = AmazonBedrock.prepare_request(:chat, model, context, opts)

      body = ReqLLM.Test.Helpers.json_body(request)
      refute Map.has_key?(body, "service_tier")
    end

    test "omits service_tier when not specified" do
      System.put_env("AWS_ACCESS_KEY_ID", "AKIATEST")
      System.put_env("AWS_SECRET_ACCESS_KEY", "secretTEST")

      {:ok, model} = ReqLLM.model("amazon-bedrock:anthropic.claude-3-haiku-20240307-v1:0")

      messages = [Context.user("Hello")]
      context = Context.new(messages)

      opts = [
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST"
      ]

      {:ok, request} = AmazonBedrock.prepare_request(:chat, model, context, opts)

      body = ReqLLM.Test.Helpers.json_body(request)
      refute Map.has_key?(body, "service_tier")
    end

    test "keeps service_tier in the InvokeModel streaming body" do
      {:ok, model} = ReqLLM.model("amazon-bedrock:anthropic.claude-3-haiku-20240307-v1:0")
      context = Context.new([Context.user("Hello")])
      opts = [access_key_id: "AKIATEST", secret_access_key: "secretTEST", service_tier: "flex"]

      {:ok, finch_request} = AmazonBedrock.attach_stream(model, context, opts, ReqLLM.Finch)

      assert finch_request.path =~ "/invoke-with-response-stream"
      body = Jason.decode!(finch_request.body)
      assert body["service_tier"] == "flex"
      refute Map.has_key?(body, "serviceTier")
    end

    test "sends serviceTier on Converse" do
      {:ok, model} = ReqLLM.model("amazon-bedrock:amazon.nova-lite-v1:0")
      context = Context.new([Context.user("Hello")])
      opts = [access_key_id: "AKIATEST", secret_access_key: "secretTEST", service_tier: "flex"]

      {:ok, request} = AmazonBedrock.prepare_request(:chat, model, context, opts)

      assert request.url.path =~ "/converse"
      body = ReqLLM.Test.Helpers.json_body(request)
      assert body["serviceTier"] == %{"type" => "flex"}
      refute Map.has_key?(body, "service_tier")
    end

    test "omits default service tiers on both Converse transports" do
      model = ReqLLM.model!("amazon_bedrock:amazon.nova-lite-v1:0")
      context = Context.new([Context.user("Hello")])

      for provider_options <- [[], [service_tier: "default"]] do
        opts = [api_key: "test-api-key", provider_options: provider_options]

        {:ok, request} = AmazonBedrock.prepare_request(:chat, model, context, opts)
        {:ok, stream} = AmazonBedrock.attach_stream(model, context, opts, ReqLLM.Finch)

        for body <- [Jason.decode!(request.body), Jason.decode!(stream.body)] do
          refute Map.has_key?(body, "serviceTier")
          refute Map.has_key?(body, "service_tier")
        end
      end
    end

    test "uses nested service tiers when explicitly routing Anthropic through Converse" do
      model = ReqLLM.model!("amazon_bedrock:anthropic.claude-3-haiku-20240307-v1:0")
      context = Context.new([Context.user("Hello")])

      for provider_options <- [
            [use_converse: true, service_tier: "priority"],
            %{"use_converse" => true, "service_tier" => "priority"},
            %{"amazon_bedrock" => %{"use_converse" => true, "service_tier" => "priority"}}
          ] do
        opts = [api_key: "test-api-key", provider_options: provider_options]

        {:ok, request} = AmazonBedrock.prepare_request(:chat, model, context, opts)
        {:ok, stream} = AmazonBedrock.attach_stream(model, context, opts, ReqLLM.Finch)

        assert request.url.path =~ "/converse"
        assert stream.path =~ "/converse-stream"

        for body <- [Jason.decode!(request.body), Jason.decode!(stream.body)] do
          assert body["serviceTier"] == %{"type" => "priority"}
          refute Map.has_key?(body, "service_tier")
        end
      end
    end

    test "sends serviceTier on ConverseStream" do
      {:ok, model} = ReqLLM.model("amazon-bedrock:amazon.nova-lite-v1:0")
      context = Context.new([Context.user("Hello")])

      opts = [
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST",
        service_tier: "reserved"
      ]

      {:ok, finch_request} = AmazonBedrock.attach_stream(model, context, opts, ReqLLM.Finch)

      assert finch_request.path =~ "/converse-stream"
      body = Jason.decode!(finch_request.body)
      assert body["serviceTier"] == %{"type" => "reserved"}
      refute Map.has_key?(body, "service_tier")
    end
  end

  describe "anthropic_beta parameter" do
    setup do
      System.put_env("AWS_ACCESS_KEY_ID", "AKIATEST")
      System.put_env("AWS_SECRET_ACCESS_KEY", "secretTEST")
      :ok
    end

    test "includes anthropic_beta in request body when provided" do
      {:ok, model} = ReqLLM.model("amazon-bedrock:anthropic.claude-3-haiku-20240307-v1:0")
      context = Context.new([Context.user("Hello")])

      opts = [
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST",
        anthropic_beta: ["output-128k-2025-02-19"]
      ]

      {:ok, request} = AmazonBedrock.prepare_request(:chat, model, context, opts)

      body = ReqLLM.Test.Helpers.json_body(request)
      assert body["anthropic_beta"] == ["output-128k-2025-02-19"]
    end

    test "omits anthropic_beta from body when not provided" do
      {:ok, model} = ReqLLM.model("amazon-bedrock:anthropic.claude-3-haiku-20240307-v1:0")
      context = Context.new([Context.user("Hello")])

      opts = [
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST"
      ]

      {:ok, request} = AmazonBedrock.prepare_request(:chat, model, context, opts)

      body = ReqLLM.Test.Helpers.json_body(request)
      refute Map.has_key?(body, "anthropic_beta")
    end

    test "supports multiple beta flags" do
      {:ok, model} = ReqLLM.model("amazon-bedrock:anthropic.claude-3-haiku-20240307-v1:0")
      context = Context.new([Context.user("Hello")])

      opts = [
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST",
        anthropic_beta: ["output-128k-2025-02-19", "another-beta-flag"]
      ]

      {:ok, request} = AmazonBedrock.prepare_request(:chat, model, context, opts)

      body = ReqLLM.Test.Helpers.json_body(request)
      assert body["anthropic_beta"] == ["output-128k-2025-02-19", "another-beta-flag"]
    end

    test "works from provider_options nesting" do
      {:ok, model} = ReqLLM.model("amazon-bedrock:anthropic.claude-3-haiku-20240307-v1:0")
      context = Context.new([Context.user("Hello")])

      opts = [
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST",
        provider_options: [anthropic_beta: ["output-128k-2025-02-19"]]
      ]

      {:ok, request} = AmazonBedrock.prepare_request(:chat, model, context, opts)

      body = ReqLLM.Test.Helpers.json_body(request)
      assert body["anthropic_beta"] == ["output-128k-2025-02-19"]
    end
  end
end
