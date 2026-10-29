defmodule ReqLLM.ProviderCaseTest do
  use ExUnit.Case, async: false

  test "restores an existing environment value exactly" do
    assert_restored(" original key\nwith whitespace ")
  end

  test "restores an empty environment value" do
    assert_restored("")
  end

  test "deletes an initially missing environment value" do
    assert_restored(nil)
  end

  test "restores environment values after a failing test body" do
    assert_restored("original-before-failure", true)
    assert_restored(nil, true)
  end

  defp assert_restored(original, fail? \\ false) do
    script = """
    ExUnit.start(autorun: false)

    defmodule CustomProvider do
      def default_env_key, do: "REQ_LLM_PROVIDER_CASE_CUSTOM_KEY"
    end

    defmodule FallbackProvider do
      def provider_id, do: :req_llm_provider_case_fallback
    end

    keys = ["REQ_LLM_PROVIDER_CASE_CUSTOM_KEY", "REQ_LLM_PROVIDER_CASE_FALLBACK_API_KEY"]

    for key <- keys do
      case #{inspect(original)} do
        nil -> System.delete_env(key)
        value -> System.put_env(key, value)
      end
    end

    defmodule CustomProviderTest do
      use ReqLLM.ProviderCase, provider: CustomProvider

      test "uses the temporary key" do
        assert System.get_env("REQ_LLM_PROVIDER_CASE_CUSTOM_KEY") == "test-key-12345"
        if #{fail?}, do: flunk("intentional ProviderCase failure")
      end
    end

    defmodule FallbackProviderTest do
      use ReqLLM.ProviderCase, provider: FallbackProvider

      test "uses the temporary key" do
        assert System.get_env("REQ_LLM_PROVIDER_CASE_FALLBACK_API_KEY") == "test-key-12345"
        if #{fail?}, do: flunk("intentional ProviderCase failure")
      end
    end

    import ExUnit.Assertions

    assert %{total: 2, failures: #{if fail?, do: 2, else: 0}} = ExUnit.run()

    for key <- keys do
      assert System.get_env(key) == #{inspect(original)}
    end
    """

    code_paths = Enum.flat_map(:code.get_path(), &["-pa", List.to_string(&1)])

    {output, status} =
      System.cmd(System.find_executable("elixir"), code_paths ++ ["-e", script],
        stderr_to_stdout: true
      )

    assert status == 0, output
    if fail?, do: assert(output =~ "intentional ProviderCase failure")
  end
end
