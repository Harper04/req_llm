defmodule ReqLLM.Test.WorkflowHTTP do
  @moduledoc false

  def request(result) do
    Req.new(url: "https://workflow.example", retry: false, adapter: __MODULE__)
    |> Req.Request.put_private(__MODULE__, result)
  end

  def run(request) do
    case Map.fetch!(request.private, __MODULE__) do
      callback when is_function(callback, 1) -> callback.(request)
      result -> {request, result}
    end
  end
end
