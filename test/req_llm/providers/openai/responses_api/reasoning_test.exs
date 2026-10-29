defmodule Provider.OpenAI.ResponsesAPI.ReasoningTest do
  use ExUnit.Case, async: false

  import ReqLLM.Test.OpenAI.ResponsesAPIHelpers, only: [build_request: 1, build_response: 2]

  alias ReqLLM.Providers.OpenAI.ResponsesAPI

  describe "encode_body/1" do
    test "encodes reasoning effort with atom" do
      request = build_request(reasoning_effort: :medium)

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert body["reasoning"] == %{"effort" => "medium"}
    end

    test "encodes reasoning effort with string" do
      request = build_request(reasoning_effort: "high")

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert body["reasoning"] == %{"effort" => "high"}
    end

    test "encodes reasoning effort :none" do
      request = build_request(reasoning_effort: :none)

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert body["reasoning"] == %{"effort" => "none"}
    end

    test "encodes reasoning effort :minimal" do
      request = build_request(reasoning_effort: :minimal)

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert body["reasoning"] == %{"effort" => "minimal"}
    end

    test "encodes reasoning effort :xhigh" do
      request = build_request(reasoning_effort: :xhigh)

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert body["reasoning"] == %{"effort" => "xhigh"}
    end

    test "encodes reasoning summary alongside effort" do
      request = build_request(reasoning_effort: :high, reasoning_summary: "auto")

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert body["reasoning"] == %{"effort" => "high", "summary" => "auto"}
    end

    test "encodes atom reasoning summary value" do
      request = build_request(reasoning_effort: :medium, reasoning_summary: :detailed)

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert body["reasoning"] == %{"effort" => "medium", "summary" => "detailed"}
    end

    test "encodes reasoning summary without effort" do
      request = build_request(reasoning_summary: "auto")

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert body["reasoning"] == %{"summary" => "auto"}
      refute Map.has_key?(body["reasoning"], "effort")
    end

    test "keeps backward-compatible reasoning shape without summary" do
      request = build_request(reasoning_effort: :low)

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert body["reasoning"] == %{"effort" => "low"}
    end

    test "omits reasoning effort when nil" do
      request = build_request(provider_options: [])

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      refute Map.has_key?(body, "reasoning")
    end

    test "passes explicit include through unchanged" do
      include = ["reasoning.encrypted_content", "file_search_call.results"]
      request = build_request(provider_options: [include: include])

      encoded = ResponsesAPI.encode_body(request)
      body = Jason.decode!(encoded.body)

      assert body["include"] == include
    end

    test "defaults include for reasoning models" do
      request = build_request(id: "gpt-5-mini")

      encoded = ResponsesAPI.encode_body(request)
      body = Jason.decode!(encoded.body)

      assert body["include"] == ["reasoning.encrypted_content"]
    end

    test "respects empty include override for reasoning models" do
      request = build_request(id: "gpt-5-mini", provider_options: [include: []])

      encoded = ResponsesAPI.encode_body(request)
      body = Jason.decode!(encoded.body)

      assert body["include"] == []
    end

    test "does not default include for non-reasoning Responses models" do
      request = build_request(id: "gpt-4o-mini")

      encoded = ResponsesAPI.encode_body(request)
      body = Jason.decode!(encoded.body)

      refute Map.has_key?(body, "include")
    end
  end

  describe "reasoning details - decode_response/1" do
    test "decodes reasoning items with summary_text array and populates reasoning_details" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output" => [
          %{
            "id" => "rs_abc123",
            "type" => "reasoning",
            "summary" => [
              %{"type" => "summary_text", "text" => "Analyzing the problem..."},
              %{"type" => "summary_text", "text" => " Breaking it down..."}
            ],
            "encrypted_content" => "base64_encrypted_content_here"
          },
          %{
            "type" => "message",
            "content" => [%{"type" => "output_text", "text" => "The answer is 42."}]
          }
        ],
        "usage" => %{"input_tokens" => 10, "output_tokens" => 50}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert %ReqLLM.Response{} = resp.body
      assert [reasoning_detail] = resp.body.message.reasoning_details
      assert %ReqLLM.Message.ReasoningDetails{} = reasoning_detail
      assert reasoning_detail.text == "Analyzing the problem...\n\n Breaking it down..."
      assert reasoning_detail.signature == "base64_encrypted_content_here"
      assert reasoning_detail.encrypted? == true
      assert reasoning_detail.provider == :openai
      assert reasoning_detail.format == "openai-responses-v1"
      assert reasoning_detail.index == 0

      assert reasoning_detail.provider_data == %{
               "id" => "rs_abc123",
               "type" => "reasoning",
               "summary" => [
                 %{"type" => "summary_text", "text" => "Analyzing the problem..."},
                 %{"type" => "summary_text", "text" => " Breaking it down..."}
               ]
             }

      assert ReqLLM.Response.thinking(resp.body) ==
               "Analyzing the problem...\n\n Breaking it down..."
    end

    test "decodes reasoning items with string summary" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output" => [
          %{
            "id" => "rs_xyz789",
            "type" => "reasoning",
            "summary" => "Thinking step by step..."
          }
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert [reasoning_detail] = resp.body.message.reasoning_details
      assert reasoning_detail.text == "Thinking step by step..."
      assert reasoning_detail.encrypted? == false
      assert reasoning_detail.signature == nil
    end

    test "decodes multiple reasoning items preserving order" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output" => [
          %{
            "id" => "rs_001",
            "type" => "reasoning",
            "summary" => [%{"type" => "summary_text", "text" => "First thought"}]
          },
          %{
            "id" => "rs_002",
            "type" => "reasoning",
            "summary" => [%{"type" => "summary_text", "text" => "Second thought"}]
          }
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert [first, second] = resp.body.message.reasoning_details
      assert first.text == "First thought"
      assert first.index == 0
      assert second.text == "Second thought"
      assert second.index == 1
    end

    test "response without reasoning items has nil reasoning_details" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output_text" => "Just a simple response",
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert resp.body.message.reasoning_details == nil
    end

    test "response with empty reasoning has nil text but retains encrypted_content" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5",
        "output" => [
          %{
            "id" => "rs_abc",
            "type" => "reasoning",
            "summary" => [],
            "encrypted_content" => "encrypted_data"
          }
        ],
        "usage" => %{"input_tokens" => 5, "output_tokens" => 10}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert [detail] = resp.body.message.reasoning_details
      assert detail.text == nil
      assert detail.signature == "encrypted_data"
      assert detail.encrypted? == true
    end
  end

  describe "reasoning details - encode_body/1" do
    test "includes reasoning items in input when no previous_response_id" do
      reasoning_detail = %ReqLLM.Message.ReasoningDetails{
        text: "Previous reasoning",
        signature: "encrypted_sig_abc",
        encrypted?: true,
        provider: :openai,
        format: "openai-responses-v1",
        index: 0,
        provider_data: %{"id" => "rs_prev123", "type" => "reasoning"}
      }

      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Previous answer"}],
        reasoning_details: [reasoning_detail]
      }

      user_msg = %ReqLLM.Message{
        role: :user,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Follow up question"}]
      }

      context = %ReqLLM.Context{messages: [assistant_msg, user_msg]}
      request = build_request(context: context)

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert [reasoning_input | _rest] = body["input"]
      assert reasoning_input["type"] == "reasoning"
      assert reasoning_input["id"] == "rs_prev123"
      assert reasoning_input["encrypted_content"] == "encrypted_sig_abc"
    end

    test "keeps each reasoning item at its own turn, so a later turn extends the input prefix" do
      reasoning = fn id ->
        %ReqLLM.Message.ReasoningDetails{
          signature: "encrypted_#{id}",
          encrypted?: true,
          provider: :openai,
          format: "openai-responses-v1",
          index: 0,
          provider_data: %{"id" => id, "type" => "reasoning"}
        }
      end

      tool_turn =
        ReqLLM.Context.assistant("",
          tool_calls: [{"read", %{"path" => "a.ex"}, [id: "call_1"]}]
        )

      first_turn = [
        ReqLLM.Context.user("Read the file"),
        %{tool_turn | reasoning_details: [reasoning.("rs_1")]},
        ReqLLM.Context.tool_result_message("read", "call_1", "file bytes", %{is_error: false}),
        %{ReqLLM.Context.assistant("Done.") | reasoning_details: [reasoning.("rs_2")]}
      ]

      shape = fn body -> Enum.map(body["input"], &(&1["type"] || &1["role"])) end

      encode = fn messages ->
        %ReqLLM.Context{messages: messages}
        |> then(&build_request(context: &1))
        |> ResponsesAPI.encode_body()
        |> ReqLLM.Test.Helpers.json_body()
      end

      before = encode.(first_turn ++ [ReqLLM.Context.user("Now edit it")])

      assert shape.(before) == [
               "user",
               "reasoning",
               "function_call",
               "function_call_output",
               "reasoning",
               "assistant",
               "user"
             ]

      assert Enum.map(before["input"], & &1["id"]) |> Enum.reject(&is_nil/1) == [
               "rs_1",
               "rs_2"
             ]

      later =
        encode.(
          first_turn ++
            [
              ReqLLM.Context.user("Now edit it"),
              %{ReqLLM.Context.assistant("Edited.") | reasoning_details: [reasoning.("rs_3")]},
              ReqLLM.Context.user("Continue")
            ]
        )

      assert Enum.take(later["input"], length(before["input"])) == before["input"]
      assert shape.(later) == shape.(before) ++ ["reasoning", "assistant", "user"]
    end

    test "does not include reasoning items when previous_response_id is present" do
      reasoning_detail = %ReqLLM.Message.ReasoningDetails{
        text: "Previous reasoning",
        signature: "encrypted_sig",
        encrypted?: true,
        provider: :openai,
        format: "openai-responses-v1",
        index: 0,
        provider_data: %{"id" => "rs_123", "type" => "reasoning"}
      }

      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Previous answer"}],
        reasoning_details: [reasoning_detail],
        metadata: %{response_id: "resp_prev_123"}
      }

      user_msg = %ReqLLM.Message{
        role: :user,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Follow up"}]
      }

      context = %ReqLLM.Context{messages: [assistant_msg, user_msg]}
      request = build_request(context: context)

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert body["previous_response_id"] == "resp_prev_123"

      refute Enum.any?(body["input"], fn item ->
               item["type"] == "reasoning"
             end)
    end

    test "store: false suppresses previous_response_id outside websocket transport" do
      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Previous answer"}],
        metadata: %{response_id: "resp_prev_123"}
      }

      user_msg = %ReqLLM.Message{
        role: :user,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Follow up"}]
      }

      context = %ReqLLM.Context{messages: [assistant_msg, user_msg]}
      request = build_request(context: context, provider_options: [store: false])

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      refute Map.has_key?(body, "previous_response_id")
      assert body["store"] == false
    end

    test "explicit previous_response_id provider option is suppressed with store: false outside websocket transport" do
      user_msg = %ReqLLM.Message{
        role: :user,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Follow up"}]
      }

      context = %ReqLLM.Context{messages: [user_msg]}

      request =
        build_request(
          context: context,
          provider_options: [store: false, previous_response_id: "resp_x"]
        )

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      refute Map.has_key?(body, "previous_response_id")
      assert body["store"] == false
    end

    test "store: true (default) preserves previous_response_id behavior" do
      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Previous answer"}],
        metadata: %{response_id: "resp_prev_456"}
      }

      user_msg = %ReqLLM.Message{
        role: :user,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Follow up"}]
      }

      context = %ReqLLM.Context{messages: [assistant_msg, user_msg]}
      request = build_request(context: context)

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert body["previous_response_id"] == "resp_prev_456"
      assert body["store"] == true
    end

    test "websocket transport with codex models defaults store to false but still chains via previous_response_id" do
      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Previous answer"}],
        metadata: %{response_id: "resp_prev_codex"}
      }

      user_msg = %ReqLLM.Message{
        role: :user,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Follow up"}]
      }

      context = %ReqLLM.Context{messages: [assistant_msg, user_msg]}

      request =
        build_request(
          id: "gpt-5.3-codex",
          context: context,
          responses_transport: :websocket
        )

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert body["previous_response_id"] == "resp_prev_codex"
      assert body["store"] == false
    end

    test "store: false without prior response_id omits both fields" do
      user_msg = %ReqLLM.Message{
        role: :user,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Hello"}]
      }

      context = %ReqLLM.Context{messages: [user_msg]}
      request = build_request(context: context, provider_options: [store: false])

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      refute Map.has_key?(body, "previous_response_id")
      assert body["store"] == false
    end

    test "skips non-OpenAI reasoning details with warning" do
      import ExUnit.CaptureLog

      anthropic_detail = %ReqLLM.Message.ReasoningDetails{
        text: "Anthropic thinking",
        signature: "anthro_sig",
        encrypted?: false,
        provider: :anthropic,
        format: "anthropic-thinking-v1",
        index: 0,
        provider_data: %{}
      }

      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Response"}],
        reasoning_details: [anthropic_detail]
      }

      user_msg = %ReqLLM.Message{
        role: :user,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Next question"}]
      }

      context = %ReqLLM.Context{messages: [assistant_msg, user_msg]}
      request = build_request(context: context)

      log =
        capture_log(fn ->
          encoded = ResponsesAPI.encode_body(request)
          body = ReqLLM.Test.Helpers.json_body(encoded)

          refute Enum.any?(body["input"], fn item ->
                   item["type"] == "reasoning"
                 end)
        end)

      assert log =~ "Skipping reasoning detail from provider :anthropic for :openai request"
    end

    test "does not replay Meta reasoning details into OpenAI requests" do
      meta_detail = %ReqLLM.Message.ReasoningDetails{
        text: "Meta reasoning",
        signature: "meta_encrypted_reasoning",
        encrypted?: true,
        provider: :meta,
        format: "openai-responses-v1",
        index: 0,
        provider_data: %{"id" => "rs_meta_1", "type" => "reasoning"}
      }

      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Response"}],
        reasoning_details: [meta_detail]
      }

      context = %ReqLLM.Context{messages: [assistant_msg]}

      body =
        build_request(context: context)
        |> ResponsesAPI.encode_body()
        |> ReqLLM.Test.Helpers.json_body()

      refute Enum.any?(body["input"], &(&1["type"] == "reasoning"))
    end

    test "encodes summary from reasoning detail text" do
      reasoning_detail = %ReqLLM.Message.ReasoningDetails{
        text: "I need to think about this carefully",
        signature: "encrypted_sig_abc",
        encrypted?: true,
        provider: :openai,
        format: "openai-responses-v1",
        index: 0,
        provider_data: %{"id" => "rs_prev123", "type" => "reasoning"}
      }

      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Answer"}],
        reasoning_details: [reasoning_detail]
      }

      user_msg = %ReqLLM.Message{
        role: :user,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Follow up"}]
      }

      context = %ReqLLM.Context{messages: [assistant_msg, user_msg]}
      request = build_request(context: context)

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      [reasoning_input | _] = body["input"]
      assert reasoning_input["type"] == "reasoning"

      assert reasoning_input["summary"] == [
               %{"type" => "summary_text", "text" => "I need to think about this carefully"}
             ]
    end

    test "encodes reasoning detail without id when provider_data has no id" do
      reasoning_detail = %ReqLLM.Message.ReasoningDetails{
        text: "Reasoning text",
        signature: "sig_123",
        encrypted?: true,
        provider: :openai,
        format: "openai-responses-v1",
        index: 0,
        provider_data: %{}
      }

      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Answer"}],
        reasoning_details: [reasoning_detail]
      }

      user_msg = %ReqLLM.Message{
        role: :user,
        content: [%ReqLLM.Message.ContentPart{type: :text, text: "Question"}]
      }

      context = %ReqLLM.Context{messages: [assistant_msg, user_msg]}
      request = build_request(context: context)

      encoded = ResponsesAPI.encode_body(request)
      body = ReqLLM.Test.Helpers.json_body(encoded)

      assert [reasoning_input | _rest] = body["input"]
      assert reasoning_input["type"] == "reasoning"
      assert reasoning_input["encrypted_content"] == "sig_123"
      refute Map.has_key?(reasoning_input, "id")
    end
  end

  describe "reasoning context and context management - encode_body/1" do
    test "encodes reasoning context alongside effort" do
      request =
        build_request(reasoning_effort: :high, provider_options: [reasoning_context: :all_turns])

      body = request |> ResponsesAPI.encode_body() |> ReqLLM.Test.Helpers.json_body()

      assert body["reasoning"] == %{"effort" => "high", "context" => "all_turns"}
    end

    test "encodes a string reasoning context without effort" do
      request = build_request(provider_options: [reasoning_context: "current_turn"])

      body = request |> ResponsesAPI.encode_body() |> ReqLLM.Test.Helpers.json_body()

      assert body["reasoning"] == %{"context" => "current_turn"}
    end

    test "encodes context_management entries from maps and keyword lists" do
      request =
        build_request(
          provider_options: [
            context_management: [
              %{type: "compaction", compact_threshold: 1000},
              [type: "truncation", strategy: "auto"]
            ]
          ]
        )

      body = request |> ResponsesAPI.encode_body() |> ReqLLM.Test.Helpers.json_body()

      assert body["context_management"] == [
               %{"type" => "compaction", "compact_threshold" => 1000},
               %{"type" => "truncation", "strategy" => "auto"}
             ]
    end

    test "omits context_management when absent or empty" do
      request = build_request(provider_options: [context_management: []])

      body = request |> ResponsesAPI.encode_body() |> ReqLLM.Test.Helpers.json_body()

      refute Map.has_key?(body, "context_management")
    end
  end
end
