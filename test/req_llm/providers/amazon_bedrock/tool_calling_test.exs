defmodule ReqLLM.Providers.AmazonBedrock.ToolCallingTest do
  use ExUnit.Case, async: true

  alias ReqLLM.{Context, Providers.AmazonBedrock}

  describe "multi-turn tool calling with Converse API" do
    test "encodes complete multi-turn conversation with tool results" do
      alias ReqLLM.ToolCall
      # Set up AWS credentials for test
      System.put_env("AWS_ACCESS_KEY_ID", "AKIATEST")
      System.put_env("AWS_SECRET_ACCESS_KEY", "secretTEST")
      System.put_env("AWS_REGION", "us-east-1")

      # Simulate a complete tool calling flow using Converse API
      {:ok, model} = ReqLLM.model("amazon-bedrock:anthropic.claude-3-haiku-20240307-v1:0")

      # Create tool call using new ToolCall API
      tool_call = ToolCall.new("toolu_add_123", "add", Jason.encode!(%{a: 5, b: 3}))

      messages = [
        Context.system("You are a calculator"),
        Context.user("What is 5 + 3?"),
        Context.assistant("I'll calculate that for you.", tool_calls: [tool_call]),
        Context.tool_result("toolu_add_123", "8")
      ]

      context = Context.new(messages)

      # Define a simple tool
      tools = [
        ReqLLM.Tool.new!(
          name: "add",
          description: "Add two numbers",
          parameter_schema: [
            a: [type: :integer, required: true],
            b: [type: :integer, required: true]
          ],
          callback: fn %{a: a, b: b} -> {:ok, a + b} end
        )
      ]

      opts = [
        tools: tools,
        use_converse: true,
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST"
      ]

      # Test that prepare_request works with tool calling
      {:ok, request} = AmazonBedrock.prepare_request(:chat, model, context, opts)

      assert %Req.Request{} = request
      # Should use Converse API endpoint when tools are present
      assert request.url.path =~ "/converse"

      # Verify the body is properly encoded
      body = ReqLLM.Test.Helpers.json_body(request)

      # Verify system instruction
      assert body["system"] == [%{"text" => "You are a calculator"}]

      # Verify messages structure
      assert is_list(body["messages"])
      assert length(body["messages"]) == 3

      [user_msg, assistant_msg, tool_result_msg] = body["messages"]

      # User message
      assert user_msg["role"] == "user"
      assert user_msg["content"] == [%{"text" => "What is 5 + 3?"}]

      # Assistant message with tool call
      assert assistant_msg["role"] == "assistant"
      assert is_list(assistant_msg["content"])
      assert length(assistant_msg["content"]) == 2

      [text_block, tool_use_block] = assistant_msg["content"]
      assert text_block["text"] == "I'll calculate that for you."
      assert tool_use_block["toolUse"]["toolUseId"] == "toolu_add_123"
      assert tool_use_block["toolUse"]["name"] == "add"
      assert tool_use_block["toolUse"]["input"] == %{"a" => 5, "b" => 3}

      # Tool result message (Converse API uses "user" role)
      assert tool_result_msg["role"] == "user",
             "Converse API requires tool results in 'user' role"

      assert is_list(tool_result_msg["content"])
      [tool_result_block] = tool_result_msg["content"]

      assert tool_result_block["toolResult"]["toolUseId"] == "toolu_add_123"
      assert tool_result_block["toolResult"]["content"] == [%{"text" => "8"}]

      # Verify toolConfig is present
      assert body["toolConfig"]
      assert is_list(body["toolConfig"]["tools"])
      assert length(body["toolConfig"]["tools"]) == 1

      [tool_spec] = body["toolConfig"]["tools"]
      assert tool_spec["toolSpec"]["name"] == "add"
    end

    test "encodes multi-turn tool calling with native Anthropic endpoint" do
      alias ReqLLM.ToolCall
      # Test with native Anthropic endpoint (not Converse API)
      System.put_env("AWS_ACCESS_KEY_ID", "AKIATEST")
      System.put_env("AWS_SECRET_ACCESS_KEY", "secretTEST")

      {:ok, model} = ReqLLM.model("amazon-bedrock:anthropic.claude-3-haiku-20240307-v1:0")

      # Create tool call using new ToolCall API
      tool_call = ToolCall.new("toolu_add_123", "add", Jason.encode!(%{a: 5, b: 3}))

      messages = [
        Context.system("You are a calculator"),
        Context.user("What is 5 + 3?"),
        Context.assistant("I'll calculate that for you.", tool_calls: [tool_call]),
        Context.tool_result("toolu_add_123", "8")
      ]

      context = Context.new(messages)

      opts = [
        use_converse: false,
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST"
      ]

      # Test that prepare_request works without forcing Converse
      {:ok, request} = AmazonBedrock.prepare_request(:chat, model, context, opts)

      # Should use native Anthropic endpoint when use_converse is false
      assert request.url.path =~ "/invoke"
      refute request.url.path =~ "/converse"

      # Verify the body uses native Anthropic format (via delegation)
      body = ReqLLM.Test.Helpers.json_body(request)

      # Native Anthropic format has different structure than Converse
      assert body["anthropic_version"] == "bedrock-2023-05-31"
      assert body["system"] == "You are a calculator"

      # Check messages are encoded with role transformation
      assert is_list(body["messages"])
      [_user_msg, _assistant_msg, tool_result_msg] = body["messages"]

      # Tool result should use "user" role (Anthropic only accepts user/assistant)
      assert tool_result_msg["role"] == "user",
             "Native Anthropic API requires tool results in 'user' role"

      # Verify tool_result content block structure
      [tool_result_block] = tool_result_msg["content"]
      assert tool_result_block["type"] == "tool_result"
      assert tool_result_block["tool_use_id"] == "toolu_add_123"
      assert tool_result_block["content"] == "8"
    end
  end
end
