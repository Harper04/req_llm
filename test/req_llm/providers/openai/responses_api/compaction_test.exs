defmodule Provider.OpenAI.ResponsesAPI.CompactionTest do
  use ExUnit.Case, async: false

  import ReqLLM.Test.OpenAI.ResponsesAPIHelpers, only: [build_request: 1, build_response: 2]

  alias ReqLLM.Providers.OpenAI.ResponsesAPI

  describe "compaction items" do
    @compaction_item %{
      "id" => "cmp_123",
      "type" => "compaction",
      "encrypted_content" => "opaque-compaction-payload"
    }

    test "decode_response/1 keeps compaction items as provider blocks on the message" do
      response_body = %{
        "id" => "resp_123",
        "model" => "gpt-5.4",
        "output" => [
          @compaction_item,
          %{"type" => "message", "content" => [%{"type" => "output_text", "text" => "Answer"}]}
        ],
        "usage" => %{"input_tokens" => 10, "output_tokens" => 5}
      }

      {_req, resp} = ResponsesAPI.decode_response(build_response(200, response_body))

      assert %ReqLLM.Response{} = resp.body
      assert ReqLLM.Response.text(resp.body) == "Answer"

      assert [%ReqLLM.Message.ContentPart{type: :provider_block} = part | _] =
               resp.body.message.content

      assert part.data == @compaction_item
      assert part.metadata.provider == :openai
      assert part.metadata.block_type == "compaction"
      assert ReqLLM.Compaction.compaction_message?(resp.body.message)

      assert [%ReqLLM.Message.ContentPart{type: :provider_block}] =
               ReqLLM.Response.provider_items(resp.body)
    end

    test "decode_stream_event/2 surfaces a completed compaction item once" do
      {:ok, model} = ReqLLM.model("openai:gpt-5")

      done_event = %{
        data: %{
          "event" => "response.output_item.done",
          "output_index" => 0,
          "item" => @compaction_item
        }
      }

      assert [chunk] = ResponsesAPI.decode_stream_event(done_event, model)
      assert chunk.type == :content_part
      assert chunk.content_part.type == :provider_block
      assert chunk.content_part.data == @compaction_item
      assert chunk.content_part.metadata.provider == :openai

      completed_event = %{
        data: %{
          "event" => "response.completed",
          "response" => %{
            "id" => "resp_123",
            "output" => [@compaction_item],
            "usage" => %{"input_tokens" => 1, "output_tokens" => 1}
          }
        }
      }

      assert [meta] = ResponsesAPI.decode_stream_event(completed_event, model)
      assert meta.type == :meta

      {:ok, response} =
        ReqLLM.Provider.Defaults.ResponseBuilder.build_response(
          [chunk, meta],
          %{id: "resp_123", model: "gpt-5"},
          context: ReqLLM.Context.new([]),
          model: model
        )

      assert [%ReqLLM.Message.ContentPart{type: :provider_block}] = response.message.content
    end

    test "encode_body/1 replays compaction blocks as top-level input items before reasoning" do
      reasoning_detail = %ReqLLM.Message.ReasoningDetails{
        text: "Plan",
        signature: "enc",
        encrypted?: true,
        provider: :openai,
        format: "openai-responses-v1",
        index: 0,
        provider_data: %{"id" => "rs_1", "type" => "reasoning"}
      }

      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [
          ReqLLM.Message.ContentPart.provider_block(:openai, @compaction_item),
          %ReqLLM.Message.ContentPart{type: :text, text: "Answer"}
        ],
        reasoning_details: [reasoning_detail]
      }

      context =
        ReqLLM.Context.new([
          assistant_msg,
          ReqLLM.Context.user("Add a booking form.")
        ])

      body =
        build_request(context: context, provider_options: [store: false])
        |> ResponsesAPI.encode_body()
        |> ReqLLM.Test.Helpers.json_body()

      assert [compaction, reasoning, assistant, user] = body["input"]
      assert compaction == @compaction_item
      assert reasoning["type"] == "reasoning"
      assert assistant["role"] == "assistant"
      assert assistant["content"] == [%{"type" => "output_text", "text" => "Answer"}]
      assert user["role"] == "user"
    end

    test "encode_body/1 drops compaction blocks owned by another provider" do
      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [
          ReqLLM.Message.ContentPart.provider_block(:anthropic, @compaction_item),
          %ReqLLM.Message.ContentPart{type: :text, text: "Answer"}
        ]
      }

      body =
        build_request(context: ReqLLM.Context.new([assistant_msg]))
        |> ResponsesAPI.encode_body()
        |> ReqLLM.Test.Helpers.json_body()

      refute Enum.any?(body["input"], &(&1["type"] == "compaction"))
    end

    test "encode_body/1 skips compaction replay when chaining via previous_response_id" do
      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [
          ReqLLM.Message.ContentPart.provider_block(:openai, @compaction_item),
          %ReqLLM.Message.ContentPart{type: :text, text: "Answer"}
        ],
        metadata: %{response_id: "resp_prev"}
      }

      body =
        build_request(context: ReqLLM.Context.new([assistant_msg, ReqLLM.Context.user("Next")]))
        |> ResponsesAPI.encode_body()
        |> ReqLLM.Test.Helpers.json_body()

      assert body["previous_response_id"] == "resp_prev"
      refute Enum.any?(body["input"], &(&1["type"] == "compaction"))
    end

    test "encode_body/1 replays Azure-owned items into Azure requests" do
      reasoning_detail = %ReqLLM.Message.ReasoningDetails{
        text: "Plan",
        signature: "enc",
        encrypted?: true,
        provider: :azure,
        format: "openai-responses-v1",
        index: 0,
        provider_data: %{"id" => "rs_1", "type" => "reasoning"}
      }

      assistant_msg = %ReqLLM.Message{
        role: :assistant,
        content: [
          ReqLLM.Message.ContentPart.provider_block(:azure, @compaction_item),
          %ReqLLM.Message.ContentPart{type: :text, text: "Answer"}
        ],
        reasoning_details: [reasoning_detail]
      }

      request = build_request(context: ReqLLM.Context.new([assistant_msg]))

      azure_request =
        %{request | options: Map.put(request.options, :req_llm_model, azure_model("gpt-5"))}

      azure_body = azure_request |> ResponsesAPI.encode_body() |> ReqLLM.Test.Helpers.json_body()
      openai_body = request |> ResponsesAPI.encode_body() |> ReqLLM.Test.Helpers.json_body()

      assert Enum.map(azure_body["input"], & &1["type"]) == ["compaction", "reasoning", nil]
      refute Enum.any?(openai_body["input"], &(&1["type"] in ["compaction", "reasoning"]))
    end
  end

  describe "build_compact_body/4" do
    test "encodes the context as input with replayed output items" do
      reasoning_detail = %ReqLLM.Message.ReasoningDetails{
        text: "Plan",
        signature: "enc",
        encrypted?: true,
        provider: :openai,
        format: "openai-responses-v1",
        index: 0,
        provider_data: %{"id" => "rs_1", "type" => "reasoning"}
      }

      context =
        ReqLLM.Context.new([
          ReqLLM.Context.user("Create a landing page."),
          %ReqLLM.Message{
            role: :assistant,
            content: [%ReqLLM.Message.ContentPart{type: :text, text: "Here it is."}],
            reasoning_details: [reasoning_detail],
            metadata: %{response_id: "resp_1"}
          }
        ])

      body = ResponsesAPI.build_compact_body(context, "gpt-5.4", provider_options: [])

      assert body["model"] == "gpt-5.4"

      assert Enum.map(body["input"], &(&1["type"] || &1["role"])) == [
               "user",
               "reasoning",
               "assistant"
             ]

      refute Map.has_key?(body, "previous_response_id")
      refute Map.has_key?(body, "stream")
      refute Map.has_key?(body, "store")
    end

    test "prefers previous_response_id over the context" do
      context = ReqLLM.Context.new([ReqLLM.Context.user("Ignored")])

      body =
        ResponsesAPI.build_compact_body(context, "gpt-5.4",
          provider_options: [previous_response_id: "resp_1"]
        )

      assert body == %{"model" => "gpt-5.4", "previous_response_id" => "resp_1"}
    end

    test "build_body/1 routes the compact operation" do
      request = build_request(context: ReqLLM.Context.new([ReqLLM.Context.user("Hi")]))
      request = %{request | options: Map.put(request.options, :operation, :compact)}

      body = request |> ResponsesAPI.encode_body() |> ReqLLM.Test.Helpers.json_body()

      assert Map.keys(body) == ["input", "model"]
    end
  end

  defp azure_model(id) do
    %LLMDB.Model{
      id: id,
      provider: :azure,
      capabilities: %{chat: true},
      extra: %{wire: %{protocol: "openai_responses"}}
    }
  end
end
