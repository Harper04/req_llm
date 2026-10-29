defmodule ReqLLM.Providers.AmazonBedrock.GuardrailsTest do
  use ExUnit.Case, async: true

  alias ReqLLM.{Context, Providers.AmazonBedrock}

  describe "native guardrail responses" do
    test "preserves guardrail assessments and normalizes interventions" do
      trace = %{"guardrail" => %{"outputs" => [%{"wordPolicy" => %{}}]}}
      request = %Req.Request{options: %{model_family: "meta"}}

      model_output = %{
        "prompt_token_count" => 10,
        "generation_token_count" => 5,
        "stop_reason" => "length"
      }

      for {action, finish_reason, output} <- [
            {"INTERVENED", :content_filter, %{}},
            {"NONE", :length, model_output}
          ] do
        body =
          Map.merge(output, %{
            "generation" => "Filtered response",
            "amazon-bedrock-guardrailAction" => action,
            "amazon-bedrock-trace" => trace
          })

        {^request, response} =
          AmazonBedrock.decode_response({request, %Req.Response{status: 200, body: body}})

        assert response.body.finish_reason == finish_reason
        assert ReqLLM.Response.text(response.body) == "Filtered response"
        assert response.body.provider_meta["amazon-bedrock-guardrailAction"] == action
        assert response.body.provider_meta["amazon-bedrock-trace"] == trace
      end
    end
  end

  describe "guardrail request configuration" do
    setup do
      {:ok, model} = ReqLLM.model("amazon-bedrock:anthropic.claude-3-haiku-20240307-v1:0")
      context = Context.new([Context.user("Hello")])

      opts = [
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST",
        region: "us-east-1",
        provider_options: [
          guardrail_identifier: "abc123",
          guardrail_version: "1",
          guardrail_trace: "enabled_full"
        ]
      ]

      {:ok, model: model, context: context, opts: opts}
    end

    test "InvokeModel puts guardrail configuration in headers", %{
      model: model,
      context: context,
      opts: opts
    } do
      {:ok, request} = AmazonBedrock.prepare_request(:chat, model, context, opts)

      assert Req.Request.get_header(request, "x-amzn-bedrock-guardrailidentifier") == ["abc123"]
      assert Req.Request.get_header(request, "x-amzn-bedrock-guardrailversion") == ["1"]
      assert Req.Request.get_header(request, "x-amzn-bedrock-trace") == ["ENABLED_FULL"]
    end

    test "Converse puts guardrail configuration in the body", %{
      model: model,
      context: context,
      opts: opts
    } do
      opts = Keyword.update!(opts, :provider_options, &Keyword.put(&1, :use_converse, true))
      {:ok, request} = AmazonBedrock.prepare_request(:chat, model, context, opts)

      assert Jason.decode!(request.body)["guardrailConfig"] == %{
               "guardrailIdentifier" => "abc123",
               "guardrailVersion" => "1",
               "trace" => "enabled_full"
             }
    end
  end
end
