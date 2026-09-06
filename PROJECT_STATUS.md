# MD visual delivery status

更新：2026-09-05

已完成：四套独立静态图和视频脚本；MatterVis 原生结构、双色键、周期盒、世界坐标向量和 DP 6 Å 邻域球；AIMD/RHF 多离子步与 SCF 密度素材；DP 64 水盒的真实能量、力、中心原子和最小镜像邻居；静态图严格 QA；DP 视频 384/384 帧 QA。完整素材、sidecar 和报告均在 `four_part_story/_qa/`。

本轮重点修订：DP 静态 PPT 图严格采用正方形周期水盒，局部 O126 的正圆 r_c 邻域直接覆盖水盒中心；右栏上半显示真实局部笛卡尔坐标和 O126 邻居距离矩阵，下半显示由真实 r_ij 计算的 radial descriptor。动画随后展示 descriptor → shared NN → Σ ε_i → E → F。数据证据写入 `_qa/04_dpmd_native/descriptor_provenance.json`；球面使用 MatterVis 原生方向光和固定相机。

还差：四套视频最终导出后的统一逐帧复核；AIMD 闪烁、停顿和离子步节奏的最终验收；文章修订稿的 XeLaTeX/XeLaTeX PDF 重编译与逐页 QA；Bohrium Notebook 全单元运行；将 metadynamics 图纳入文章最终排版。

当前输出：

- `four_part_story/figures/01_velocity_verlet.png` 至 `04_deep_potential_md.png`
- `four_part_story/videos/01_velocity_verlet.mp4` 至 `04_deep_potential_md.mp4`
- `four_part_story/_qa/04_dpmd_native/qa_report_strict.json`
- `four_part_story/_qa/04_dpmd_native/_qa/every_frame_qa.json`
- `four_part_story/figures/05_well_tempered_metadynamics.png`
- `four_part_story/videos/05_well_tempered_metadynamics.mp4`
- `four_part_story/_qa/05_metadynamics/qa_report_strict.json`
- `four_part_story/figures/06_rigid_water_descriptor_invariance.png`
- `four_part_story/videos/06_rigid_water_descriptor_invariance.mp4`
- `four_part_story/_qa/06_symmetry_invariance/descriptor_provenance.json`
