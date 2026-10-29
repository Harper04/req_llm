defmodule ReqLLM.TestSupport.ApplicationConfigTest do
  use ExUnit.Case, async: false

  for outcome <- [:passing, :failing] do
    @outcome outcome

    test "restores present, nil, false and absent values after a #{outcome} test" do
      script = """
      ExUnit.start(autorun: false)

      Application.put_env(:req_llm, :config_helper_present, :original)
      Application.put_env(:req_llm, :config_helper_nil, nil)
      Application.put_env(:req_llm, :config_helper_false, false)
      Application.delete_env(:req_llm, :config_helper_absent)

      defmodule ApplicationConfigFixture do
        use ExUnit.Case, async: false

        test "temporary configuration" do
          ReqLLM.TestSupport.ApplicationConfig.put_env(
            config_helper_present: :temporary,
            config_helper_nil: :temporary,
            config_helper_false: :temporary,
            config_helper_absent: :temporary
          )

          for key <- [:config_helper_present, :config_helper_nil,
                      :config_helper_false, :config_helper_absent] do
            assert Application.fetch_env(:req_llm, key) == {:ok, :temporary}
          end

          ReqLLM.TestSupport.ApplicationConfig.put_env(config_helper_present: :second)
          assert Application.fetch_env(:req_llm, :config_helper_present) == {:ok, :second}
          #{if @outcome == :failing, do: ~s|flunk("intentional failure")|}
        end
      end

      %{total: 1, failures: #{if @outcome == :failing, do: 1, else: 0}} = ExUnit.run()
      {:ok, :original} = Application.fetch_env(:req_llm, :config_helper_present)
      {:ok, nil} = Application.fetch_env(:req_llm, :config_helper_nil)
      {:ok, false} = Application.fetch_env(:req_llm, :config_helper_false)
      :error = Application.fetch_env(:req_llm, :config_helper_absent)
      IO.puts("configuration restored")
      """

      {output, status} =
        System.cmd(
          System.find_executable("elixir"),
          ["-r", Path.expand("application_config.ex", __DIR__), "-e", script],
          stderr_to_stdout: true
        )

      assert status == 0, output
      assert output =~ "configuration restored"
    end
  end
end
