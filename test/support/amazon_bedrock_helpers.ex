defmodule ReqLLM.Test.AmazonBedrockHelpers do
  @moduledoc false

  def sparse_adaptive_opus do
    %LLMDB.Model{
      id: "anthropic.claude-opus-4-7-v1:0",
      model: "anthropic.claude-opus-4-7-v1:0",
      provider: :amazon_bedrock,
      provider_model_id: "us.anthropic.claude-opus-4-7-v1:0",
      capabilities: %{chat: true, reasoning: %{enabled: true}},
      extra: %{family: "claude-opus"}
    }
  end

  def bedrock_opts(extra) do
    Keyword.merge(
      [
        access_key_id: "AKIATEST",
        secret_access_key: "secretTEST",
        region: "us-east-1",
        max_tokens: 2000
      ],
      extra
    )
  end

  def build_aws_event_stream_message(payload) when is_binary(payload) do
    headers = <<>>
    headers_length = byte_size(headers)
    payload_length = byte_size(payload)
    total_length = 16 + headers_length + payload_length

    prelude = <<total_length::32-big, headers_length::32-big>>
    prelude_crc = :erlang.crc32(prelude)

    message_without_crc = <<
      prelude::binary,
      prelude_crc::32,
      headers::binary,
      payload::binary
    >>

    message_crc = :erlang.crc32(message_without_crc)

    <<
      total_length::32-big,
      headers_length::32-big,
      prelude_crc::32,
      headers::binary,
      payload::binary,
      message_crc::32
    >>
  end
end
