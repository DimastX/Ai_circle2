@echo off
echo Testing VSL scenarios...

echo.
echo Testing wave scenario...
python src/run_circle_simulation.py --config-name 1k --max-num-vehicles 22 --vsl --vsl-scenario wave --T-prime 1.0 --simulation-duration 30 --sumo-tools-dir "C:\Program Files (x86)\Eclipse\sumo-1.22.0\tools" --output-dir results/test_vsl_wave

echo.
echo Testing throughput scenario...
python src/run_circle_simulation.py --config-name 1k --max-num-vehicles 22 --vsl --vsl-scenario throughput --T-prime 0.7 --simulation-duration 30 --sumo-tools-dir "C:\Program Files (x86)\Eclipse\sumo-1.22.0\tools" --output-dir results/test_vsl_throughput

echo.
echo Testing variance scenario...
python src/run_circle_simulation.py --config-name 1k --max-num-vehicles 22 --vsl --vsl-scenario variance --T-prime 1.1 --simulation-duration 30 --sumo-tools-dir "C:\Program Files (x86)\Eclipse\sumo-1.22.0\tools" --output-dir results/test_vsl_variance

echo.
echo All tests completed! 