# Scripts

Entry-point placeholders for a future public release. **Do not dump GB-scale data here.**

Live runners currently sit in the research tree, for example:

```text
/home/wael/Test_Code/tools/mmdetection3d/evaluate_weather_phy.py
/home/wael/Test_Code/tools/mmdetection3d/evaluate_weather_quality.py
/home/wael/Test_Code/tools/mmdetection3d/test_all_validation_weather.sh
/home/wael/Test_Code/tools/mmdetection3d/test_fusion_validation_weather.sh
/home/wael/Test_Code/tools/mmdetection3d/run_test_weather.sh
/home/wael/Test_Code/weathergen/generate.py
/home/wael/Test_Code/weathergen/generate_all_weather.py
/home/wael/Test_Code/weathergen/evaluate_weather.py
```

When publishing, copy only slim wrappers that accept a `data_root` / `checkpoint` argument and document download steps in the root README.
