# 文章修订记录（md-article staging）

日期：2026-09-04

## 改动范围

本轮只修改了 report/md_reading_notes.tex，并新增本记录文件；没有修改 four_part_story、three_slide_story 或任何视觉脚本，也没有提交或推送。

主要改动：

1. 在初值问题后加入两个短例框。其一直接读取 four_part_story/data/vv_h2o_step.npz 的真实首步（\(\Delta t=0.5\) fs、氧原子 \(x\) 分量的位置/速度/加速度和能量变化）；其二是明确标注为“非模拟结果”的一维谐振子量纲算例（\(q_0=1.0\) Å、\(v_0=0.2\) Å/fs、\(a_0=-0.5\) Å/fs²、\(\Delta t=0.1\) fs）。
2. 明确显式 Euler、辛 Euler 和 velocity Verlet 的适用边界，并把表 1 限定为保守、可分 Hamiltonian 的固定步长 NVE 条件；修正 shadow Hamiltonian 能量表述。
3. 在 AIMD 段注明图示数据为中性闭壳层 RHF/STO-3G 水二聚体，核力来自收敛 RHF 能量的三维中心有限差分，区分 Fock 矩阵和核力记号。
4. 将图 3 的 \(e_E^{(k)}=|E^{(k)}-E^\star|/E_h\) 明确标为能量距离而非 SCF 残差，补充密度矩阵更新残差定义、\(10^{-8}E_h\) 能量停止条件及真实 SCF 计数/残差范围。
5. 说明密度等值线来自固定平面/网格上的真实 AO 密度，Gaussian blur 只是残差驱动的视觉编码；补充 Deep Potential 局域到全局求和、示例盒参数和原子能不可独立物理解读的限定。
6. 在采样段加入真实七个连续 AIMD 几何的描述性统计（均值、样本标准差、范围，并明确不能据此给 \(N_{\rm eff}\)）；另加标注为非本文数据的 \(N=1000,\Delta t=1\) fs、\(\tau_{\rm int}=20\) fs \(\Rightarrow N_{\rm eff}=25\) 教学算例。将首现术语补上中英文桥接，并删去“最小数值自查清单”式元话语。

## 编译/验证

尝试在 report/ 下用 XeLaTeX 编译：

xelatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf md_reading_notes.tex

本机 MiKTeX 在初始化阶段返回 Windows API error 5（拒绝访问），并且无法写入其日志目录；在临时 MIKTEX_USERCONFIG/MIKTEX_USERDATA 目录重试仍同样失败。进程已在短时限内停止，因此本轮没有可确认的新 PDF 页数或渲染结果；待整合环境修复 MiKTeX 权限后需重新运行 XeLaTeX/Biber 并逐页检查版式。源文件中的新增 quote、equation 环境已做文本配对检查。
