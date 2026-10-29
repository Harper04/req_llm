defmodule ReqLLM.Test.HTTPAdapter do
  @moduledoc false

  def options(callback) do
    [plugins: [&attach(&1, callback)]]
  end

  def attach(request, callback) do
    request
    |> Map.put(:adapter, __MODULE__)
    |> Req.Request.put_private(__MODULE__, callback)
  end

  def run(request) do
    callback = Req.Request.get_private(request, __MODULE__)
    callback.(request)
  end
end
