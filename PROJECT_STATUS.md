# MD visual delivery status

更新：2026-09-05

已完成：四套独立静态图和视频脚本；MatterVis 原生结构、双色键、周期盒、世界坐标向量和 DP 6 Å 邻域球；AIMD/RHF 多离子步与 SCF 密度素材；DP 64 水盒的真实能量、力、中心原子和最小镜像邻居；静态图严格 QA；DP 视频 384/384 帧 QA。完整素材、sidecar 和报告均在 `four_part_story/_qa/`。

本轮重点修订：DP 球体以 O126 为世界坐标球心，固定斜视相机下保持正圆，使用 MatterVis CPU 原生方向光和高光表现前后半球，并保留球内真实水分子、MIC 向量和中心力箭头。球面没有采用二维椭圆或纸面遮罩。

还差：四套视频最终导出后的统一逐帧复核；AIMD 闪烁、停顿和离子步节奏的最终验收；well-tempered metadynamics Demo；文章修订稿的 XeLaTeX/XeLaTeX PDF 重编译与逐页 QA；Bohrium Notebook 全单元运行；MatterVis 分支到 `main` 的 PR 收尾。

当前输出：

- `four_part_story/figures/01_velocity_verlet.png` 至 `04_deep_potential_md.png`
- `four_part_story/videos/01_velocity_verlet.mp4` 至 `04_deep_potential_md.mp4`
- `four_part_story/_qa/04_dpmd_native/qa_report_strict.json`
- `four_part_story/_qa/04_dpmd_native/_qa/every_frame_qa.json`
