defmodule ReqLLM.RuntimeTest do
  use ExUnit.Case, async: true

  alias ReqLLM.RequestPlan
  alias ReqLLM.Runtime

  defmodule Provider do
    def prepare_request(operation, model, input, opts) do
      send(opts[:owner], {:prepared, operation, model, input, opts})
      {:ok, opts[:request]}
    end
  end

  test "executes the provider's Req pipeline before applying the operation result policy" do
    model = %LLMDB.Model{provider: :runtime_test, id: "outside-the-catalog"}
    input = <<0, 1, 255>>
    schema = NimbleOptions.new!(value: [type: :string])

    request =
      Req.new(
        url: "https://runtime.example",
        plug: fn conn ->
          assert Plug.Conn.get_req_header(conn, "x-provider") == ["prepared"]
          Plug.Conn.send_resp(conn, 202, "wire response")
        end
      )
      |> Req.Request.append_request_steps(
        provider_encode: &Req.Request.put_header(&1, "x-provider", "prepared")
      )
      |> Req.Request.append_response_steps(
        provider_decode: fn {request, response} ->
          {request, %{response | body: {:decoded, response.body}}}
        end
      )

    opts = [owner: self(), request: request, schema: schema, duplicate: 1, duplicate: 2]
    assert {:ok, plan} = RequestPlan.from_resolved(model, :ocr, Provider, opts, :req)
    result_ref = make_ref()

    policy = fn %Req.Response{status: 202, body: {:decoded, "wire response"}} ->
      {:ok, result_ref}
    end

    assert Runtime.execute(plan, input, :infinity, policy) == {:ok, result_ref}
    assert_receive {:prepared, :ocr, ^model, ^input, ^opts}
    refute_received {:prepared, _, _, _, _}
  end
end
