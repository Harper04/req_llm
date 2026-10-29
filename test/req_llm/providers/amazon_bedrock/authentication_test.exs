defmodule ReqLLM.Providers.AmazonBedrock.AuthenticationTest do
  use ExUnit.Case, async: true

  alias ReqLLM.{Context, Providers.AmazonBedrock}

  describe "AWS session token support" do
    test "includes session token in signed Req request headers" do
      # Test non-streaming path (Req pipeline via put_aws_sigv4)
      {:ok, model} = ReqLLM.model("amazon-bedrock:anthropic.claude-3-haiku-20240307-v1:0")
      context = Context.new([Context.user("Hello")])
      session_token = "FwoGZXIvYXdzEBYaDHhBTEMPLESessionToken123"

      opts = [
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST",
        region: "us-east-1",
        session_token: session_token
      ]

      # Build request using prepare_request
      {:ok, request} = AmazonBedrock.prepare_request(:chat, model, context, opts)

      # The request should have the AWS SigV4 signing step attached
      assert %Req.Request{} = request

      # Manually trigger the request steps to verify signing behavior
      # The aws_sigv4 step should add the session token header
      signed_request =
        request.request_steps
        |> Enum.reduce(request, fn {_name, step}, req ->
          step.(req)
        end)

      # Check that signed headers include session token
      headers_map =
        Map.new(signed_request.headers, fn
          {k, [v | _]} -> {String.downcase(k), v}
          {k, v} -> {String.downcase(k), v}
        end)

      assert headers_map["x-amz-security-token"] == session_token

      # Verify proper AWS4-HMAC-SHA256 signature with token
      auth_header = headers_map["authorization"]
      assert auth_header =~ "AWS4-HMAC-SHA256"
      assert auth_header =~ "x-amz-security-token"
    end
  end
end
