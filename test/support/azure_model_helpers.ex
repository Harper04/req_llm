defmodule ReqLLM.Test.AzureModelHelpers do
  @moduledoc false

  def traditional_openai_model do
    %LLMDB.Model{
      id: "gpt-4o",
      provider: :azure,
      capabilities: %{chat: true},
      extra: %{}
    }
  end
end
