defmodule ReqLLM.Test.AnthropicToolHelpers do
  @moduledoc false

  import ReqLLM.ProviderCase, only: [context_fixture: 0]

  alias ReqLLM.Providers.Anthropic

  def deferred_tool(name) do
    ReqLLM.Tool.new!(
      name: name,
      description: "A deferred tool",
      parameter_schema: [name: [type: :string, required: true]],
      callback: fn _ -> {:ok, "result"} end,
      provider_options: [anthropic: [defer_loading: true]]
    )
  end

  def direct_tool(name) do
    ReqLLM.Tool.new!(
      name: name,
      description: "An always-loaded tool",
      parameter_schema: [name: [type: :string, required: true]],
      callback: fn _ -> {:ok, "result"} end
    )
  end

  # `anthropic_prompt_cache` is read from the request options, where the
  # option pipeline flattens it; the search config is a provider option.
  def encode_tools(tools, provider_options, request_options \\ []) do
    {:ok, model} = ReqLLM.model("anthropic:claude-sonnet-4-6")

    %Req.Request{
      options:
        [
          context: context_fixture(),
          model: model.model,
          stream: false,
          tools: tools,
          provider_options: provider_options
        ] ++ request_options
    }
    |> Anthropic.encode_body()
    |> ReqLLM.Test.Helpers.json_body()
    |> Map.fetch!("tools")
  end
end
