defmodule ReqLLM.Test.OpenAI.ResponsesAPIHelpers do
  @moduledoc """
  Shared request, response, and event builders for OpenAI Responses API unit tests.
  """

  alias ReqLLM.Providers.OpenAI.ResponsesAPI

  def phased_message_item(id, phase, text) do
    %{
      "id" => id,
      "type" => "message",
      "role" => "assistant",
      "status" => "completed",
      "content" => [%{"type" => "output_text", "text" => text, "annotations" => []}]
    }
    |> then(fn item -> if phase, do: Map.put(item, "phase", phase), else: item end)
  end

  def refusal_message_item(id, refusal) do
    %{
      "id" => id,
      "type" => "message",
      "role" => "assistant",
      "status" => "completed",
      "content" => [%{"type" => "refusal", "refusal" => refusal}]
    }
  end

  def decode_stream_events(events, model) do
    {chunks, _state} =
      Enum.flat_map_reduce(events, nil, fn data, state ->
        ResponsesAPI.decode_stream_event(%{data: data}, model, state)
      end)

    chunks
  end

  def build_request(opts) do
    context = Keyword.get(opts, :context, %ReqLLM.Context{messages: []})
    provider_opts = Keyword.get(opts, :provider_options, [])

    provider_opts =
      case Keyword.get(opts, :reasoning_summary) do
        nil -> provider_opts
        summary -> Keyword.put(provider_opts, :reasoning_summary, summary)
      end

    req_opts =
      %{
        id: Keyword.get(opts, :id, "gpt-5"),
        context: context,
        stream: Keyword.get(opts, :stream),
        max_output_tokens: Keyword.get(opts, :max_output_tokens),
        max_completion_tokens: Keyword.get(opts, :max_completion_tokens),
        max_tokens: Keyword.get(opts, :max_tokens),
        tools: Keyword.get(opts, :tools),
        tool_choice: Keyword.get(opts, :tool_choice),
        parallel_tool_calls: Keyword.get(opts, :parallel_tool_calls),
        reasoning_effort: Keyword.get(opts, :reasoning_effort),
        provider_options: provider_opts
      }
      |> maybe_put_responses_transport(Keyword.get(opts, :responses_transport))

    %Req.Request{
      method: :post,
      url: URI.parse("https://api.openai.com/v1/responses"),
      headers: %{},
      body: {:json, %{}},
      options: req_opts
    }
  end

  defp maybe_put_responses_transport(opts, nil), do: opts

  defp maybe_put_responses_transport(opts, transport),
    do: Map.put(opts, :responses_transport, transport)

  def build_response(status, body, opts \\ []) do
    context = Keyword.get(opts, :context, %ReqLLM.Context{messages: []})

    req = %Req.Request{
      method: :post,
      url: URI.parse("https://api.openai.com/v1/responses"),
      headers: %{},
      body: {:json, %{}},
      options: %{id: "gpt-5", context: context}
    }

    resp = %Req.Response{
      status: status,
      headers: %{},
      body: body
    }

    {req, resp}
  end
end
