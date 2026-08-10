-- Use the exact same logical source slice for all three POCs.

-- 1) Smoke test query:
SELECT TOP 10 *
FROM <DATABASE>.<TABLE>;

-- 2) Source row-count for the benchmark slice:
SELECT COUNT(*) AS expected_rows
FROM (
    <PASTE_BENCHMARK_SELECT_HERE>
) q;

-- Keep the benchmark SELECT deterministic and avoid SELECT * if practical.
-- Record the expected_rows before each benchmark if the source is mutable.