import unittest

from scripts.run_modal import (
    ADAPTIVE_SELECTOR_KEYS_ENV,
    BLOCK_SHAPE_TUNING_ENV,
    COMPOSITE_COMPILED_HARNESS_ENV,
    COMPOSITE_REFERENCE_HARNESS_ENV,
    EXPERIMENTAL_BLOCK_POLICY_ENV,
    PERSISTENT_POLICY_ENV,
    build_runtime_env,
)


class RunModalHelpersTest(unittest.TestCase):
    def test_build_runtime_env_includes_adaptive_selector_keys(self) -> None:
        runtime_env = build_runtime_env(
            persistent_policy="adaptive",
            adaptive_selector_keys="key-a,key-b",
        )

        self.assertEqual(
            runtime_env,
            {
                PERSISTENT_POLICY_ENV: "adaptive",
                ADAPTIVE_SELECTOR_KEYS_ENV: "key-a,key-b",
            },
        )

    def test_build_runtime_env_includes_block_policy_flags(self) -> None:
        runtime_env = build_runtime_env(
            experimental_block_policy=True,
            block_shape_tuning=True,
            composite_reference_harness=True,
            composite_compiled_harness=True,
        )

        self.assertEqual(
            runtime_env,
            {
                EXPERIMENTAL_BLOCK_POLICY_ENV: "1",
                BLOCK_SHAPE_TUNING_ENV: "1",
                COMPOSITE_REFERENCE_HARNESS_ENV: "1",
                COMPOSITE_COMPILED_HARNESS_ENV: "1",
            },
        )



if __name__ == "__main__":
    unittest.main()
