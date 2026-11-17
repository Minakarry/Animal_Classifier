"""Build clean Word drafts with real tables, embedded figures, and placeholders."""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "output" / "figures"


def set_run_font(run, east_asia: str = "宋体", western: str = "Times New Roman", size: int = 12):
    run.font.name = western
    run.font.size = Pt(size)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), east_asia)


def add_para(doc, text="", *, bold=False, italic=False, size=12, center=False, east="宋体", space_after=6):
    p = doc.add_paragraph()
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(space_after)
    if text:
        run = p.add_run(text)
        run.bold = bold
        run.italic = italic
        set_run_font(run, east_asia=east, size=size)
    return p


def add_heading_cn(doc, text, level):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        set_run_font(run, east_asia="黑体", size=16 if level == 0 else (14 if level == 1 else 12))
    return h


def add_heading_en(doc, text, level):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        set_run_font(run, east_asia="Times New Roman", western="Times New Roman", size=16 if level == 0 else (14 if level == 1 else 12))
    return h


def add_table(doc, headers, rows, col_widths=None):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    for j, h in enumerate(headers):
        cell = table.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        run.bold = True
        set_run_font(run, size=10)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = table.rows[i + 1].cells[j]
            cell.text = ""
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(str(val))
            set_run_font(run, size=10)
    if col_widths:
        for row in table.rows:
            for j, w in enumerate(col_widths):
                row.cells[j].width = Cm(w)
    doc.add_paragraph()
    return table


def add_figure(doc, path: Path, caption: str, width_inches=5.8):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if path.exists():
        run = p.add_run()
        run.add_picture(str(path), width=Inches(width_inches))
    else:
        run = p.add_run(f"[Figure file missing: {path.name}]")
        run.font.color.rgb = RGBColor(180, 0, 0)
        set_run_font(run, size=11)
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(10)
    r = cap.add_run(caption)
    r.bold = True
    set_run_font(r, size=10)


def add_placeholder(doc, caption: str, hint: str):
    box = doc.add_paragraph()
    box.alignment = WD_ALIGN_PARAGRAPH.CENTER
    box.paragraph_format.space_before = Pt(8)
    box.paragraph_format.space_after = Pt(2)
    run = box.add_run("【此处插入图片 / Insert figure here】")
    run.bold = True
    run.font.color.rgb = RGBColor(120, 120, 120)
    set_run_font(run, size=11)

    hint_p = doc.add_paragraph()
    hint_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    hr = hint_p.add_run(hint)
    hr.italic = True
    hr.font.color.rgb = RGBColor(120, 120, 120)
    set_run_font(hr, size=9)

    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(10)
    cr = cap.add_run(caption)
    cr.bold = True
    set_run_font(cr, size=10)


CLASS_ROWS = [
    ["Butterfly", "0.99", "0.96", "0.97", "212"],
    ["Cat", "0.97", "0.96", "0.96", "167"],
    ["Chicken", "0.99", "0.97", "0.98", "310"],
    ["Cow", "0.96", "0.95", "0.95", "187"],
    ["Dog", "0.95", "0.97", "0.96", "487"],
    ["Elephant", "0.97", "0.97", "0.97", "145"],
    ["Horse", "0.96", "0.97", "0.97", "263"],
    ["Sheep", "0.97", "0.96", "0.96", "182"],
    ["Spider", "0.98", "0.99", "0.99", "483"],
    ["Squirrel", "0.97", "0.97", "0.97", "187"],
]

FULL_REFS = [
    '[1] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “ImageNet: A large-scale hierarchical image database,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2009, pp. 248–255.',
    '[2] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classification with deep convolutional neural networks,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2012.',
    '[3] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 770–778.',
    '[4] A. Corrado, “Animals-10 dataset,” Kaggle, 2018. [Online]. Available: https://www.kaggle.com/datasets/alessiocorrado99/animals10',
    '[5] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 618–626.',
    '[6] K. Simonyan and A. Zisserman, “Very deep convolutional networks for large-scale image recognition,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2015.',
    '[7] C. Szegedy et al., “Going deeper with convolutions,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2015, pp. 1–9.',
    '[8] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards real-time object detection with region proposal networks,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2015.',
    '[9] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look once: Unified, real-time object detection,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 779–788.',
    '[10] A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in Adv. Neural Inf. Process. Syst. (NeurIPS), 2019.',
    '[11] TorchVision maintainers and contributors, “TorchVision: PyTorch’s computer vision library.” [Online]. Available: https://pytorch.org/vision/',
    '[12] Riverbank Computing, “PyQt5 reference guide.” [Online]. Available: https://www.riverbankcomputing.com/static/Docs/PyQt5/',
    '[13] Pallets Projects, “Flask documentation.” [Online]. Available: https://flask.palletsprojects.com/',
]


def build_zh(out_path: Path):
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.8)
    section.right_margin = Cm(2.8)

    add_heading_cn(doc, "基于 ResNet18 的动物图像分类系统设计与实现", 0)
    add_para(doc, "研究生小组项目报告", bold=True, center=True, size=14)
    add_para(doc, "小组成员：[请填写姓名]", center=True)
    add_para(doc, "单位：[请填写学校 / 学院 / 专业]", center=True)
    add_para(doc, "通讯作者 / 组长：[请填写]", center=True)
    add_para(doc, "日期：[请填写]", center=True, space_after=16)

    add_heading_cn(doc, "摘要", 1)
    add_para(
        doc,
        "准确的动物图像分类是计算机视觉中的代表性任务，在科普教育、课程项目与轻量交互原型中具有明确的演示价值。"
        "本研究生小组项目并不面向工业级野外监测部署，而是致力于构建一条完整、可复现、可解释的识别流水线，"
        "使非专业用户可通过桌面端与网页端进行试用。具体而言，我们在公开基准 Animals-10 上对 ImageNet 预训练的 ResNet18 进行微调，"
        "以识别十类常见动物。除模型训练外，还实现了涵盖数据准备、可复现评估、推理、PyQt5 桌面客户端与 Flask Web 服务的软件原型。"
        "界面支持上传后自动识别、Top-3 预测、低置信度提示、Grad-CAM 热力图、类别说明与历史记录管理。"
        "在留出测试集上的实验表明，微调模型取得了较强的 Top-1 / Top-3 准确率与较为均衡的各类 F1。"
        "进一步在匹配协议下开展 ImageNet 预训练微调与从头训练的对照消融，以支持迁移学习的有效性。"
        "文中还讨论了失败案例以及整图单标签分类的局限。本工作的意义不在于提出全新网络结构，"
        "而在于将经典深度学习方法集成为可部署系统，服务于研究生阶段的协作科研训练。",
    )
    add_para(doc, "关键词：动物图像分类；ResNet18；迁移学习；Animals-10；消融实验；Grad-CAM；Top-K 预测；深度学习系统", italic=True)

    add_heading_cn(doc, "1. 引言", 1)
    add_heading_cn(doc, "1.1 背景与动机", 2)
    add_para(
        doc,
        "图像分类仍是视觉识别的核心任务之一。借助 ImageNet [1] 等大规模标注数据集与 GPU 算力，自 AlexNet [2] 以来，"
        "卷积神经网络（CNN）已成为视觉理解的主流范式。动物图像分类是一个便于教学、内容充实的应用方向："
        "既包含姿态变化、复杂背景与类间相似等挑战，又可将类别空间控制在小组项目可管理的范围内。"
        "本项目的典型用户是查看演示的任课教师、探索已训模型的学生，或只想上传照片并立刻得到可理解结果的普通用户"
        "——而非生产级生态监测系统的运维人员。",
    )
    add_para(
        doc,
        "基于手工特征（如 SIFT/HOG）与浅层分类器的传统流程，对无约束动物照片往往泛化不佳。"
        "深度残差网络（ResNet）[3] 通过跳跃连接缓解深层优化困难，至今仍是强有力的视觉骨干。"
        "迁移学习通过将 ImageNet 预训练表征适配到 Animals-10 基准 [4]，进一步降低数据与算力成本。"
        "对交互式演示而言，仅给出单一硬标签并不够：Top-K 备选、低置信度提示与 Grad-CAM 解释 [5] "
        "有助于用户判断何时可信、模型大致关注了何处。",
    )
    add_para(
        doc,
        "项目意义。本工作价值体现在三方面：（1）应用——面向教育与轻量内容交互的可用识别原型；"
        "（2）方法——在匹配协议下证明 ImageNet 预训练优于从头训练；"
        "（3）系统/工程——从数据准备、评估到双端部署与基础可解释性的完整闭环。"
        "我们不宣称提出新网络结构，而强调交付可复现、可演示的研究生科研成果。",
    )

    add_heading_cn(doc, "1.2 问题定义", 2)
    add_para(
        doc,
        "给定一张 RGB 输入图像，系统需预测十类动物中的一类，返回置信度，给出 Top-3 假设，并为预测类可视化 Grad-CAM 证据。"
        "训练好的模型还需封装为面向非专业用户的交互式桌面与 Web 应用。挑战因此包含两方面："
        "（i）在 Animals-10 上通过迁移学习获得可靠分类性能；"
        "（ii）以适合课程演示的可维护软件栈暴露推理、不确定提示与轻量可解释性。",
    )

    add_heading_cn(doc, "1.3 主要贡献", 2)
    for t in [
        "建立面向 Animals-10 的 ResNet18 可复现微调流水线，包括数据组织、增强、固定种子训练、早停，以及按验证准确率选择最优模型。",
        "设计并实现双端识别系统（PyQt5 桌面 + Flask 网页），支持 Top-3 展示、低置信度提示、Grad-CAM 热力图、类别说明与历史管理。",
        "提供专用评估脚本，输出 Top-1 / Top-3 准确率、宏平均 / 加权 F1、逐类指标、混淆矩阵与各类 F1 图。",
        "在匹配训练协议下开展预训练与从头训练消融，量化 ImageNet 初始化的贡献。",
        "对识别行为做定性分析，包括分布外失败案例以及单标签分类的固有局限。",
    ]:
        p = doc.add_paragraph(t, style="List Number")
        for run in p.runs:
            set_run_font(run)

    add_heading_cn(doc, "1.4 论文结构", 2)
    add_para(
        doc,
        "第 2 节综述相关工作。第 3 节介绍数据集与预处理。第 4 节给出模型与方法。第 5 节说明系统实现。"
        "第 6 节报告实验结果与讨论。第 7 节总结并展望未来工作。",
    )

    add_heading_cn(doc, "2. 相关工作", 1)
    add_heading_cn(doc, "2.1 深度卷积网络结构", 2)
    add_para(
        doc,
        "早期深度 CNN 成功之后，VGG [6]、Inception（GoogLeNet）[7] 等结构进一步提升了 ImageNet 识别性能，"
        "ResNet 则成为广泛使用的残差骨干。综合考虑交互应用中的精度与时延折中，本项目采用 ResNet18 作为骨干网络。",
    )
    add_heading_cn(doc, "2.2 面向域适配的迁移学习", 2)
    add_para(
        doc,
        "当目标数据规模中等时，迁移学习尤为有效。通过 ImageNet 预训练权重初始化并替换最终分类层，"
        "模型可快速适配新的标签空间。本文对 ResNet18 采用骨干与分类头差分学习率进行微调。",
    )
    add_heading_cn(doc, "2.3 可解释性与置信度感知界面", 2)
    add_para(
        doc,
        "实用识别系统受益于排序备选（Top-K）、不确定情形的置信度阈值，以及 Grad-CAM 等视觉归因方法。"
        "本原型已将上述能力集成到桌面端与网页端。",
    )
    add_heading_cn(doc, "2.4 动物识别系统", 2)
    add_para(
        doc,
        "许多研究原型只关注离线精度，很少交付完整面向用户的系统。本项目强调从数据准备到部署评估的一体化流水线。"
        "Faster R-CNN [8]、YOLO [9] 等检测器更适合多实例定位；本组目标选择整图级分类作为可控任务。",
    )

    add_heading_cn(doc, "3. 数据集与预处理", 1)
    add_heading_cn(doc, "3.1 Animals-10 基准", 2)
    add_para(doc, "Animals-10 约含 28,000 张、覆盖十类动物的图像。图像来自网络并经人工检查，更接近实际噪声条件。")
    add_para(doc, "表 1  Animals-10 类别列表", bold=True, center=True, size=10)
    add_table(
        doc,
        ["编号", "英文类名", "中文类名", "界面显示名"],
        [
            ["0", "butterfly", "蝴蝶", "Butterfly"],
            ["1", "cat", "猫", "Cat"],
            ["2", "chicken", "鸡", "Chicken"],
            ["3", "cow", "牛", "Cow"],
            ["4", "dog", "狗", "Dog"],
            ["5", "elephant", "象", "Elephant"],
            ["6", "horse", "马", "Horse"],
            ["7", "sheep", "羊", "Sheep"],
            ["8", "spider", "蜘蛛", "Spider"],
            ["9", "squirrel", "松鼠", "Squirrel"],
        ],
        col_widths=[2, 3.5, 3, 3.5],
    )

    add_heading_cn(doc, "3.2 数据组织与划分协议", 2)
    add_para(
        doc,
        "图像按 ImageFolder 层次存放于 data/<class>/。对每一类以固定种子（SEED=42）随机打乱，并按约 9:1 划分训练/测试子集。"
        "划分列表保存为 train.txt 与 test.txt。本机重新生成后包含 23,556 条训练样本与 2,623 条测试样本。",
    )
    add_heading_cn(doc, "3.3 预处理、归一化与数据增强", 2)
    add_para(
        doc,
        "图像缩放为 224×224，并使用 ImageNet 均值/方差归一化。训练增强包括随机水平翻转与颜色抖动；"
        "验证与推理变换为确定性变换。",
    )

    add_heading_cn(doc, "4. 方法", 1)
    add_heading_cn(doc, "4.1 模型形式化", 2)
    add_para(doc, "设输入 RGB 图像为 x，其真实类别标签为")
    add_para(doc, "y ∈ {0, 1, …, 9}.", center=True, italic=True)
    add_para(doc, "分类器为参数化网络 fθ(·)，将 x 映射为 10 维 logit 向量")
    add_para(doc, "z = fθ(x) ∈ ℝ¹⁰,", center=True, italic=True)
    add_para(doc, "其中 θ 表示全部可训练参数。类别概率由")
    add_para(doc, "p = softmax(z),    p ∈ ℝ¹⁰,    Σᵢ pᵢ = 1", center=True, italic=True)
    add_para(doc, "得到。预测标签为")
    add_para(doc, "ŷ = argmaxᵢ pᵢ.", center=True, italic=True)
    add_para(
        doc,
        "训练最小化 p 与 y 的 one-hot 编码之间的交叉熵损失。"
        "我们使用 ImageNet 初始化的 ResNet18，并将最后的全连接层替换为 10 类线性分类器。",
    )

    add_heading_cn(doc, "4.2 优化策略", 2)
    for t in [
        "优化器：Adam",
        "学习率：骨干 1×10⁻⁴，分类头 1×10⁻³（为骨干的 10 倍）",
        "调度器：余弦退火；余弦周期等于设定的训练轮数",
        "默认设置：30 个 epoch，批大小 32，随机种子 42",
        "早停：验证准确率连续 8 个 epoch 无提升则停止",
        "检查点：验证准确率最优权重保存至 output/resnet18_best.pth",
    ]:
        p = doc.add_paragraph(t, style="List Bullet")
        for run in p.runs:
            set_run_font(run)

    add_heading_cn(doc, "4.3 推理：Top-K、低置信度与 Grad-CAM", 2)
    add_para(doc, "上传图像经验证变换预处理后送入网络。由 softmax 概率向量 p，系统返回：")
    for t in [
        "Top-1 类别 ŷ、其置信度 pŷ 以及文字说明；",
        "Top-3 类别–置信度对（p 中最大的三个分量）；",
        "当 Top-1 置信度低于 0.50 时给出低置信度警告；",
        "在 ResNet18 最后残差阶段（layer4）上计算 Grad-CAM 热力图。",
    ]:
        p = doc.add_paragraph(t, style="List Number")
        for run in p.runs:
            set_run_font(run)
    add_para(doc, "桌面端与网页端共享同一推理后端（predict.py）。")

    add_heading_cn(doc, "4.4 离线评估协议", 2)
    add_para(doc, "evaluate.py 加载最优检查点，遍历 test.txt，导出 metrics.json、混淆矩阵与各类 F1 图。")

    add_heading_cn(doc, "4.5 预训练与从头训练消融", 2)
    add_para(doc, "为隔离迁移学习效果，compare.py 在完全相同的协议下训练两个 ResNet18：")
    add_para(doc, "表 2  消融实验设置", bold=True, center=True, size=10)
    add_table(
        doc,
        ["设置", "初始化", "轮数", "其他条件"],
        [
            ["预训练", "ImageNet 权重", "10", "与另一组一致"],
            ["从头训练", "随机 / 无 ImageNet 权重", "10", "与另一组一致"],
        ],
        col_widths=[3, 4.5, 2, 3.5],
    )
    add_para(
        doc,
        "共享控制：随机种子 42，Adam 骨干学习率 1×10⁻⁴、头学习率 1×10⁻³，余弦退火，批大小 32，相同数据划分。"
        "消融采用 10 轮预算以便公平对比；部署系统仍使用全量微调最优检查点。",
    )

    add_heading_cn(doc, "5. 系统设计与实现", 1)
    add_heading_cn(doc, "5.1 总体架构", 2)
    add_para(
        doc,
        "系统分为三层：数据/模型层、推理层、应用层（桌面 + 网页）。两端界面以英文为主。"
        "深度学习栈基于 PyTorch [10] 与 TorchVision [11]；界面分别用 PyQt5 [12] 与 Flask [13] 实现。",
    )
    add_figure(doc, FIG / "system_architecture.png", "图 1  系统架构概览", width_inches=6.2)

    add_heading_cn(doc, "5.2 模块职责", 2)
    add_para(doc, "表 3  主要模块与职责", bold=True, center=True, size=10)
    add_table(
        doc,
        ["模块", "职责"],
        [
            ["download_dataset.py", "获取并重组 Animals-10"],
            ["generate_dataset.py", "构造可复现划分列表"],
            ["utils.py", "数据集抽象与变换"],
            ["seed_utils.py", "全局随机种子"],
            ["train.py", "微调、早停与检查点"],
            ["evaluate.py", "测试指标与评估图"],
            ["compare.py", "预训练 vs 从头消融"],
            ["predict.py", "Top-K / 不确定提示 / Grad-CAM"],
            ["window.py", "PyQt5 桌面客户端"],
            ["myflask.py", "Flask Web 服务"],
            ["config.py", "全局配置与类别元数据"],
        ],
        col_widths=[5, 10],
    )

    add_heading_cn(doc, "5.3 桌面与网页界面", 2)
    add_para(doc, "两端均支持：图像上传与自动识别；Top-1 结果、置信度与类别说明；Top-3 概率条；置信度 < 50% 警告；Grad-CAM；历史记录管理。")
    add_placeholder(doc, "图 3  桌面端界面截图（含 Top-3、Grad-CAM、低置信度提示）", "请运行 window.py 后截图插入")
    add_placeholder(doc, "图 4  网页端界面截图", "请运行 myflask.py 后截图插入")

    add_heading_cn(doc, "5.4 范围与非目标", 2)
    add_para(
        doc,
        "本原型面向课程演示与交互探索，做整图单标签分类，不定位多动物。"
        "工业相机陷阱流程、强开集拒识与多目标检测不在范围内。",
    )

    add_heading_cn(doc, "6. 实验与讨论", 1)
    add_heading_cn(doc, "6.1 实验设置", 2)
    for t in [
        "软件：Python、PyTorch / TorchVision、scikit-learn、matplotlib、Pillow、PyQt5、Flask",
        "数据划分：23,556 训练 / 2,623 测试（固定种子，约 9:1）",
        "主评估检查点：output/resnet18_best.pth",
        "消融：python compare.py --epochs 10",
        "硬件：AMD Ryzen 7 5800H, NVIDIA GeForce RTX 3050 Laptop GPU",
    ]:
        p = doc.add_paragraph(t, style="List Bullet")
        for run in p.runs:
            set_run_font(run)

    add_figure(
        doc,
        FIG / "training_curves.png",
        "图 2  训练损失与验证准确率曲线（消融：预训练 vs 从头训练，各 10 个 epoch）",
        width_inches=5.8,
    )
    add_para(
        doc,
        "说明：主部署模型的完整 epoch 级训练日志未单独导出；图 2 基于消融实验（compare.py）保存的逐轮 train_loss / val_acc 绘制，"
        "用于展示匹配协议下预训练与从头训练的收敛差异。",
    )

    add_heading_cn(doc, "6.2 定量结果（测试集）", 2)
    add_para(doc, "evaluate.py 在 2,623 张测试图像上的结果如下。")
    add_para(doc, "表 4  总体测试指标", bold=True, center=True, size=10)
    add_table(
        doc,
        ["指标", "数值"],
        [
            ["Top-1 准确率", "97.03%"],
            ["Top-3 准确率", "99.24%"],
            ["宏平均 F1", "0.9686"],
            ["加权 F1", "0.9703"],
        ],
        col_widths=[6, 4],
    )

    add_para(doc, "表 5  各类精确率 / 召回率 / F1（测试集）", bold=True, center=True, size=10)
    add_table(
        doc,
        ["类别", "精确率", "召回率", "F1", "样本数"],
        CLASS_ROWS,
        col_widths=[3, 2.5, 2.5, 2.5, 2.5],
    )

    add_figure(doc, FIG / "confusion_matrix.png", "图 5  测试集混淆矩阵")
    add_figure(doc, FIG / "per_class_f1.png", "图 6  各类 F1 分数柱状图")

    add_heading_cn(doc, "6.3 消融：预训练微调 vs 从头训练", 2)
    add_para(
        doc,
        "使用 compare.py --epochs 10，在数据、种子、优化器、学习率与轮数完全相同的条件下对比。"
        "下表数值来自 output/compare_results.json。",
    )
    add_para(doc, "表 6  预训练与从头训练对比", bold=True, center=True, size=10)
    add_table(
        doc,
        ["设置", "最佳验证准确率", "训练时间"],
        [
            ["预训练（ImageNet 初始化）", "96.99%", "2688.7 s（约 44.8 分钟）"],
            ["从头训练", "80.75%", "2388.6 s（约 39.8 分钟）"],
        ],
        col_widths=[5, 4.5, 3.5],
    )
    add_para(
        doc,
        "解读：匹配的 10 轮预算下，预训练微调明显优于从头训练（96.99% vs 80.75%），支持 ImageNet 特征向 Animals-10 有效迁移。"
        "消融绝对值低于第 6.2 节，因为仅训练 10 个 epoch，而部署模型来自全量微调。",
    )
    add_figure(doc, FIG / "compare_accuracy.png", "图 8  预训练 vs 从头训练准确率对比")

    add_heading_cn(doc, "6.4 定性分析", 2)
    add_para(
        doc,
        "除总体指标外，我们从测试集中选取典型样本做定性观察（示例图见 images/demo_for_paper，"
        "并汇总为图 7）。",
    )
    for t in [
        "成功案例：主体突出、背景中等复杂时，Top-1 置信度常接近 100%，Grad-CAM 热力图集中在动物身体/头部，说明决策依据与人类直觉一致。",
        "Top-3 有竞争的案例：例如羊与鸡等外观/背景干扰较强时，Top-1 仍可能正确，但 Top-2 可占约 10%–25% 概率，体现备选列表的实用价值。",
        "失败案例：偶发“关注区域大致正确、类别却判错”的情况（如犬类被高置信度判为其他类）。这说明 Grad-CAM 只能解释空间关注，不能保证类别正确；论文截图中宜同时展示成功与失败例。",
        "低置信度 / 分布外：非动物、严重遮挡或目标过小时，系统可触发 <50% 警告，提示用户勿过度信任 Top-1。",
        "多动物照片：整图单标签设定下只返回一个全局类别，无法对应各个体；此类需求更适合检测框架。",
    ]:
        p = doc.add_paragraph(t, style="List Bullet")
        for run in p.runs:
            set_run_font(run)
    add_figure(doc, FIG / "gradcam_examples.png", "图 7  定性示例：成功 / Top-3 模糊 / 失败（左：原图，右：Grad-CAM）", width_inches=5.6)

    add_heading_cn(doc, "6.5 讨论", 2)
    add_para(
        doc,
        "在教育 / 演示向识别原型的范围内，ResNet18 迁移学习有效且足以支撑交互使用。"
        "第 6.2 节表明部署模型测试精度较强（Top-1 97.03%，Top-3 99.24%）；第 6.3 节在匹配协议下进一步隔离了 ImageNet 初始化贡献"
        "（预训练 96.99% vs 从头训练 80.75%）。"
        "Top-3、置信度阈值与 Grad-CAM 提升了可用性与可解释性，但也暴露出高置信度误判仍可能出现。"
        "局限包括封闭十类体系、softmax 置信度校准不足、消融轮数较短，以及无法定位多目标。"
        "截图素材可使用 images/demo_for_paper 中的成功 / Top-3 / 失败样本；桌面与网页界面截图（图 3、图 4）需在运行 window.py / myflask.py 后自行插入。",
    )

    add_heading_cn(doc, "7. 结论与展望", 1)
    add_para(
        doc,
        "本项目交付了基于 ResNet18 与 Animals-10 的完整动物图像分类系统，覆盖迁移学习、双端部署、"
        "离线评估、预训练/从头消融与轻量可解释性。测试集上全量微调系统达到 Top-1 97.03%、Top-3 99.24%。"
        "未来可探索多目标检测、更强开集拒识、更长多种子消融与模型压缩。",
    )

    add_heading_cn(doc, "致谢", 1)
    add_para(doc, "作者感谢任课教师与全体组员在讨论与协作开发中的支持。")

    add_heading_cn(doc, "参考文献", 1)
    for r in FULL_REFS:
        add_para(doc, r, size=10, space_after=3)

    add_heading_cn(doc, "附录 C  小组贡献说明", 1)
    add_table(
        doc,
        ["成员", "角色", "主要贡献", "估计工作量（%）"],
        [
            ["[姓名]", "组长 / 算法", "[填写]", "[ ]"],
            ["[姓名]", "系统 / GUI", "[填写]", "[ ]"],
            ["[姓名]", "网页 / 实验 / 撰写", "[填写]", "[ ]"],
        ],
        col_widths=[3, 3.5, 5, 3.5],
    )

    doc.save(out_path)
    print("saved", out_path)


def build_en(out_path: Path):
    """English draft kept in lockstep with build_zh (same sections / tables / figures)."""
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.8)
    section.right_margin = Cm(2.8)
    ea = "Times New Roman"

    add_heading_en(doc, "Design and Implementation of an Animal Image Classification System Based on ResNet18", 0)
    add_para(doc, "Graduate Group Project Report", bold=True, center=True, size=14, east=ea)
    for t in [
        "Group Members: [Please fill in names]",
        "Affiliation: [Please fill in university / school / program]",
        "Corresponding Author / Group Leader: [Please fill in]",
        "Date: [Please fill in]",
    ]:
        add_para(doc, t, center=True, east=ea)

    add_heading_en(doc, "Abstract", 1)
    add_para(
        doc,
        "Accurate animal image classification is a representative computer-vision task with clear demonstration value "
        "for science education, course projects, and lightweight interactive prototypes. Rather than targeting industrial "
        "wildlife-monitoring deployment, this graduate group project focuses on building a complete, reproducible, and "
        "explainable recognition pipeline that non-expert users can try through desktop and web interfaces. Specifically, "
        "a ResNet18 model pretrained on ImageNet is fine-tuned on the public Animals-10 benchmark to recognize ten common "
        "animal categories. Beyond model training, we engineer a software prototype covering dataset preparation, "
        "reproducible evaluation, inference, a PyQt5 desktop client, and a Flask web service. The interfaces provide "
        "automatic recognition after image uploading, Top-3 predictions, low-confidence warnings, Grad-CAM heatmaps, "
        "category descriptions, and history management. Experiments on the held-out test set show that the fine-tuned "
        "model attains strong Top-1 / Top-3 accuracy and balanced per-class F1 scores. A controlled ablation further "
        "compares ImageNet-pretrained fine-tuning with training from scratch under a matched protocol, supporting the "
        "benefit of transfer learning. We also discuss failure modes and the limitation of single-label whole-image "
        "classification. The significance of this work lies less in proposing a new architecture than in integrating "
        "classical deep-learning methods into a deployable system for graduate-level collaborative research training.",
        east=ea,
    )
    add_para(
        doc,
        "Keywords: Animal Image Classification; ResNet18; Transfer Learning; Animals-10; Ablation Study; Grad-CAM; "
        "Top-K Prediction; Deep Learning System; Graduate Group Project",
        italic=True,
        east=ea,
    )

    add_heading_en(doc, "1. Introduction", 1)
    add_heading_en(doc, "1.1 Background and Motivation", 2)
    add_para(
        doc,
        "Image classification remains one of the core tasks in visual recognition. With large-scale annotated datasets "
        "such as ImageNet [1] and GPU computing, convolutional neural networks (CNNs) have become the dominant paradigm "
        "since AlexNet [2]. Animal image classification is a convenient and pedagogically rich target domain: it involves "
        "pose variation, cluttered backgrounds, and inter-class similarity, yet the label space can be kept manageable "
        "for a group project. Typical end users in our setting are course instructors reviewing a demo, students exploring "
        "a trained model, or general users who simply upload a photo and want an immediate, understandable answer—not "
        "operators of a production ecological-monitoring system.",
        east=ea,
    )
    add_para(
        doc,
        "Traditional pipelines based on handcrafted descriptors (e.g., SIFT/HOG) and shallow classifiers often generalize "
        "poorly to unconstrained animal photographs. Deep residual networks (ResNet) [3] alleviate optimization difficulties "
        "via skip connections and remain strong visual backbones. Transfer learning further reduces data and compute cost "
        "by adapting ImageNet-pretrained representations to the Animals-10 benchmark [4]. For interactive demos, a single "
        "hard label is insufficient: Top-K alternatives, low-confidence warnings, and Grad-CAM explanations [5] help users "
        "decide when to trust the model and roughly where it looked.",
        east=ea,
    )
    add_para(
        doc,
        "Project significance. The value of this work is threefold: (1) application—a usable recognition prototype for "
        "education and lightweight content interaction; (2) methodology—empirical evidence that ImageNet pretraining "
        "outperforms training from scratch under a matched protocol; (3) system/engineering—a full loop from data "
        "preparation and evaluation to dual-interface deployment with basic interpretability. We do not claim a novel "
        "network architecture; we claim a coherent graduate research artifact that is reproducible and demonstrable.",
        east=ea,
    )

    add_heading_en(doc, "1.2 Problem Statement", 2)
    add_para(
        doc,
        "Given an input RGB image, the system predicts one of ten animal categories, returns a confidence score, exposes "
        "Top-3 hypotheses, and visualizes Grad-CAM evidence for the predicted class. The trained model must also be "
        "packaged as interactive desktop and web applications for non-expert users. The challenge is therefore twofold: "
        "(i) obtain reliable classification performance via transfer learning on Animals-10; and (ii) expose inference, "
        "uncertainty cues, and lightweight interpretability through a maintainable software stack suitable for course demonstration.",
        east=ea,
    )

    add_heading_en(doc, "1.3 Contributions", 2)
    for t in [
        "We establish a reproducible fine-tuning pipeline for ResNet18 on Animals-10, including dataset organization, augmentation, seeded training, early stopping, and best-model selection based on validation accuracy.",
        "We design and implement a dual-interface recognition system (PyQt5 desktop + Flask web) with Top-3 display, low-confidence warnings, Grad-CAM heatmaps, species descriptions, and history management.",
        "We provide a dedicated evaluation script that reports Top-1 / Top-3 accuracy, macro / weighted F1, per-class metrics, a confusion matrix, and per-class F1 figures.",
        "We conduct a pretrained vs. from-scratch ablation under a matched training protocol to quantify the contribution of ImageNet initialization.",
        "We analyze recognition behavior qualitatively, including out-of-distribution failures and the inherent limitation of single-label classification.",
    ]:
        p = doc.add_paragraph(t, style="List Number")
        for run in p.runs:
            set_run_font(run, east_asia=ea)

    add_heading_en(doc, "1.4 Paper Organization", 2)
    add_para(
        doc,
        "Section 2 reviews related work. Section 3 describes the dataset and preprocessing. Section 4 presents the model "
        "and methods. Section 5 details system implementation. Section 6 reports experiments and discussion. Section 7 "
        "concludes and outlines future work.",
        east=ea,
    )

    add_heading_en(doc, "2. Related Work", 1)
    add_heading_en(doc, "2.1 Deep Convolutional Architectures", 2)
    add_para(
        doc,
        "Following the success of early deep CNNs, architectures such as VGG [6] and Inception (GoogLeNet) [7] further "
        "improved ImageNet recognition performance, while ResNet became a widely used residual backbone. Considering the "
        "trade-off between accuracy and latency for interactive applications, we adopt ResNet18 as the backbone of this project.",
        east=ea,
    )
    add_heading_en(doc, "2.2 Transfer Learning for Domain Adaptation", 2)
    add_para(
        doc,
        "Transfer learning is particularly effective when target data are moderate in scale. By initializing from "
        "ImageNet-pretrained weights and replacing the final classifier, models can rapidly adapt to new label spaces. "
        "Our approach fine-tunes ResNet18 with differential learning rates for backbone and classification head.",
        east=ea,
    )
    add_heading_en(doc, "2.3 Interpretability and Confidence-Aware Interfaces", 2)
    add_para(
        doc,
        "Practical recognition systems benefit from ranked alternatives (Top-K), confidence thresholds for uncertain cases, "
        "and visual attribution methods such as Grad-CAM. Our prototype integrates these features into both desktop and web clients.",
        east=ea,
    )
    add_heading_en(doc, "2.4 Animal Recognition Systems", 2)
    add_para(
        doc,
        "Many research prototypes focus only on offline accuracy and rarely deliver complete user-facing systems. This "
        "project emphasizes an integrated pipeline from data preparation to deployment and evaluation. Object detectors "
        "such as Faster R-CNN [8] and YOLO [9] are more appropriate for multi-instance localization; for this group "
        "objective we target image-level classification as a controlled research task.",
        east=ea,
    )

    add_heading_en(doc, "3. Dataset and Preprocessing", 1)
    add_heading_en(doc, "3.1 Animals-10 Benchmark", 2)
    add_para(
        doc,
        "Animals-10 contains approximately 28,000 animal images across ten categories. Images were collected from web "
        "sources and manually inspected, making the benchmark closer to practical noisy conditions.",
        east=ea,
    )
    add_para(doc, "Table 1  Animals-10 class list", bold=True, center=True, size=10, east=ea)
    add_table(
        doc,
        ["ID", "Class (EN)", "Class (ZH)", "UI display name"],
        [
            ["0", "butterfly", "蝴蝶", "Butterfly"],
            ["1", "cat", "猫", "Cat"],
            ["2", "chicken", "鸡", "Chicken"],
            ["3", "cow", "牛", "Cow"],
            ["4", "dog", "狗", "Dog"],
            ["5", "elephant", "象", "Elephant"],
            ["6", "horse", "马", "Horse"],
            ["7", "sheep", "羊", "Sheep"],
            ["8", "spider", "蜘蛛", "Spider"],
            ["9", "squirrel", "松鼠", "Squirrel"],
        ],
        col_widths=[2, 3.5, 3, 3.5],
    )

    add_heading_en(doc, "3.2 Data Organization and Split Protocol", 2)
    add_para(
        doc,
        "Images are organized in an ImageFolder hierarchy under data/<class>/. For each class, samples are randomly "
        "shuffled with a fixed seed (SEED=42) and partitioned into training/testing subsets at approximately 9:1. Split "
        "lists are saved as train.txt and test.txt. After regeneration, the lists contain 23,556 training samples and "
        "2,623 test samples.",
        east=ea,
    )
    add_heading_en(doc, "3.3 Preprocessing, Normalization, and Augmentation", 2)
    add_para(
        doc,
        "Images are resized to 224×224 and normalized with ImageNet mean/std statistics. Training augmentation includes "
        "random horizontal flipping and color jittering; validation and inference transforms are deterministic.",
        east=ea,
    )

    add_heading_en(doc, "4. Methodology", 1)
    add_heading_en(doc, "4.1 Model Formulation", 2)
    add_para(doc, "Let the input RGB image be x, and its ground-truth class label be", east=ea)
    add_para(doc, "y ∈ {0, 1, …, 9}.", center=True, italic=True, east=ea)
    add_para(doc, "The classifier is a parameterized network fθ(·). It maps x to a 10-dimensional logit vector", east=ea)
    add_para(doc, "z = fθ(x) ∈ ℝ¹⁰,", center=True, italic=True, east=ea)
    add_para(doc, "where θ denotes all trainable parameters. Class probabilities are obtained by", east=ea)
    add_para(doc, "p = softmax(z),    p ∈ ℝ¹⁰,    Σᵢ pᵢ = 1", center=True, italic=True, east=ea)
    add_para(doc, "The predicted label is", east=ea)
    add_para(doc, "ŷ = argmaxᵢ pᵢ.", center=True, italic=True, east=ea)
    add_para(
        doc,
        "Training minimizes the cross-entropy loss between p and the one-hot encoding of y. We use ResNet18 initialized "
        "from ImageNet; the final fully connected layer is replaced by a 10-way linear classifier. In train.py, pretrained "
        "weights are loaded unless --no-pretrained is set; model.fc is then replaced by a linear layer with NUM_CLASSES "
        "outputs and Xavier initialization.",
        east=ea,
    )
    add_placeholder(
        doc,
        "Code Fig. 1  ResNet18 loading, classifier-head replacement, and cross-entropy loss (train.py)",
        "Screenshot train.py lines ~106–115",
    )

    add_heading_en(doc, "4.2 Optimization Strategy", 2)
    add_para(
        doc,
        "Training hyperparameters are centralized in config.py, including image size, batch size, epochs, learning rate, "
        "train/test split ratio, random seed, confidence threshold, and early-stopping patience, which improves reproducibility.",
        east=ea,
    )
    add_placeholder(
        doc,
        "Code Fig. 2  Global training and inference hyperparameters (config.py)",
        "Screenshot config.py lines ~75–82",
    )
    for t in [
        "Optimizer: Adam",
        "Learning rates: backbone 1×10⁻⁴, classification head 1×10⁻³ (10× the backbone rate)",
        "Scheduler: cosine annealing; the cosine period equals the number of requested epochs",
        "Default schedule: 30 epochs, batch size 32, random seed 42",
        "Early stopping: stop if validation accuracy does not improve for 8 consecutive epochs",
        "Checkpoint: the best validation-accuracy weights are saved to output/resnet18_best.pth for deployment",
    ]:
        p = doc.add_paragraph(t, style="List Bullet")
        for run in p.runs:
            set_run_font(run, east_asia=ea)
    add_para(
        doc,
        "Data preprocessing and augmentation are implemented in LoadData (utils.py). Training uses random horizontal flip "
        "and color jitter; validation/testing use only resize and ImageNet mean/std normalization, avoiding stochasticity "
        "at evaluation time.",
        east=ea,
    )
    add_placeholder(
        doc,
        "Code Fig. 3  Training augmentation and validation preprocessing pipelines (utils.py)",
        "Screenshot utils.py lines ~21–32",
    )

    add_heading_en(doc, "4.3 Inference: Top-K, Low Confidence, and Grad-CAM", 2)
    add_para(
        doc,
        "Uploaded images are preprocessed with validation transforms and forwarded through the network. From the softmax "
        "probability vector p, the system returns:",
        east=ea,
    )
    for t in [
        "Top-1 class ŷ, its confidence pŷ, and a textual description;",
        "Top-3 class–confidence pairs (the three largest entries of p);",
        "A low-confidence warning when the Top-1 confidence is below 0.50;",
        "A Grad-CAM heatmap on the last residual stage of ResNet18 (layer4).",
    ]:
        p = doc.add_paragraph(t, style="List Number")
        for run in p.runs:
            set_run_font(run, east_asia=ea)
    add_para(
        doc,
        "In predict.py, softmax probabilities are ranked to form a Top-3 list. If the Top-1 confidence is below "
        "CONFIDENCE_THRESHOLD (default 0.50), uncertain is set to true and a low-confidence message is returned. Desktop "
        "and web clients share the same prediction backend (predict.py: predict_topk / predict_full / generate_gradcam).",
        east=ea,
    )
    add_placeholder(
        doc,
        "Code Fig. 4  Top-3 inference and low-confidence decision (predict.py)",
        "Screenshot predict.py lines ~91–115",
    )
    add_para(
        doc,
        "For interpretability, Grad-CAM is computed on ResNet18 layer4. Forward/backward hooks collect activations and "
        "gradients; back-propagation on the target-class score yields a heatmap overlaid on the input, showing where the "
        "model attends for the predicted class.",
        east=ea,
    )
    add_placeholder(
        doc,
        "Code Fig. 5  Grad-CAM hook registration on layer4 (predict.py)",
        "Screenshot predict.py lines ~132–155",
    )

    add_heading_en(doc, "4.4 Offline Evaluation Protocol", 2)
    add_para(
        doc,
        "evaluate.py loads the best checkpoint, iterates over test.txt, and exports metrics.json, a confusion matrix, "
        "and per-class F1 figures.",
        east=ea,
    )

    add_heading_en(doc, "4.5 Pretrained vs. From-Scratch Ablation", 2)
    add_para(
        doc,
        "To isolate the effect of transfer learning, compare.py trains two ResNet18 models under an identical protocol:",
        east=ea,
    )
    add_para(doc, "Table 2  Ablation settings", bold=True, center=True, size=10, east=ea)
    add_table(
        doc,
        ["Setting", "Initialization", "Epochs", "Other conditions"],
        [
            ["Pretrained", "ImageNet weights", "10", "Matched with the other run"],
            ["From-Scratch", "Random / no ImageNet weights", "10", "Matched with the other run"],
        ],
        col_widths=[3, 4.5, 2, 3.5],
    )
    add_para(
        doc,
        "Shared controls: random seed 42, Adam with backbone LR 1×10⁻⁴ and head LR 1×10⁻³, cosine annealing, batch size 32, "
        "and the same data split. Model construction for the ablation is handled by build_model in compare.py: ImageNet "
        "weights are loaded when pretrained=True; otherwise training starts without pretrained weights. The 10-class head "
        "and the rest of the training protocol remain matched, isolating the effect of initialization.",
        east=ea,
    )
    add_placeholder(
        doc,
        "Code Fig. 6  Pretrained vs. from-scratch model construction for ablation (compare.py)",
        "Screenshot compare.py lines ~39–44 (build_model)",
    )
    add_para(
        doc,
        "The ablation uses a 10-epoch budget for a fair comparison; the deployed system still uses the fully fine-tuned "
        "best checkpoint.",
        east=ea,
    )

    add_heading_en(doc, "5. System Design and Implementation", 1)
    add_heading_en(doc, "5.1 Architecture Overview", 2)
    add_para(
        doc,
        "The system has three layers: data/model, inference, and application (desktop + web). English is used as the "
        "primary UI language. The deep-learning stack is built on PyTorch [10] and TorchVision [11]; interfaces are "
        "implemented with PyQt5 [12] and Flask [13].",
        east=ea,
    )
    add_figure(doc, FIG / "system_architecture.png", "Figure 1  System architecture overview", width_inches=6.2)

    add_heading_en(doc, "5.2 Module Responsibilities", 2)
    add_para(doc, "Table 3  Main modules and responsibilities", bold=True, center=True, size=10, east=ea)
    add_table(
        doc,
        ["Module", "Responsibility"],
        [
            ["download_dataset.py", "Acquire and reorganize Animals-10"],
            ["generate_dataset.py", "Build reproducible train/test lists"],
            ["utils.py", "Dataset abstraction and transforms"],
            ["seed_utils.py", "Global seeding"],
            ["train.py", "Fine-tuning, early stopping, checkpoints"],
            ["evaluate.py", "Test metrics and evaluation figures"],
            ["compare.py", "Pretrained vs from-scratch ablation"],
            ["predict.py", "Top-K / uncertainty / Grad-CAM"],
            ["window.py", "PyQt5 desktop client"],
            ["myflask.py", "Flask web service"],
            ["config.py", "Global configs and class metadata"],
        ],
        col_widths=[5, 10],
    )
    add_para(
        doc,
        "Modules follow the pipeline data preparation → training/evaluation → unified inference → dual UI. config.py holds "
        "globals; utils.py and train.py handle data and fine-tuning; evaluate.py / compare.py produce offline metrics and "
        "the ablation; predict.py supplies Top-3, low-confidence flags, and Grad-CAM to window.py (desktop) and myflask.py "
        "(web). The UI layer focuses on interaction and display; recognition logic is not duplicated in the clients, which "
        "reduces inconsistency between desktop and web results. Runtime screenshots of the GUIs are preferred over long UI "
        "source listings.",
        east=ea,
    )

    add_heading_en(doc, "5.3 Desktop and Web Interfaces", 2)
    add_para(
        doc,
        "Both clients support image upload and automatic recognition; Top-1 result with confidence and description; "
        "Top-3 probability bars; a low-confidence warning when confidence < 50%; Grad-CAM visualization; and history management.",
        east=ea,
    )
    add_placeholder(doc, "Figure 3  Desktop GUI screenshot (Top-3 + Grad-CAM + low-confidence warning)", "Run window.py and insert screenshot")
    add_placeholder(doc, "Figure 4  Web interface screenshot", "Run myflask.py and insert screenshot")

    add_heading_en(doc, "5.4 Scope and Non-Goals", 2)
    add_para(
        doc,
        "This prototype targets course demonstration and interactive exploration. It performs whole-image single-label "
        "classification and does not localize multiple animals. Industrial camera-trap workflows, strong open-set rejection, "
        "and multi-object detection are out of scope.",
        east=ea,
    )

    add_heading_en(doc, "6. Experiments and Discussion", 1)
    add_heading_en(doc, "6.1 Experimental Setup", 2)
    for t in [
        "Software: Python, PyTorch / TorchVision, scikit-learn, matplotlib, Pillow, PyQt5, Flask",
        "Data split: 23,556 train / 2,623 test (fixed seed, ~9:1)",
        "Main evaluation checkpoint: output/resnet18_best.pth",
        "Ablation: python compare.py --epochs 10",
        "Hardware: AMD Ryzen 7 5800H, NVIDIA GeForce RTX 3050 Laptop GPU",
    ]:
        p = doc.add_paragraph(t, style="List Bullet")
        for run in p.runs:
            set_run_font(run, east_asia=ea)
    add_figure(
        doc,
        FIG / "training_curves.png",
        "Figure 2  Training loss and validation accuracy curves (ablation: pretrained vs from-scratch, 10 epochs each)",
        width_inches=5.8,
    )
    add_para(
        doc,
        "Note: a full epoch-wise log for the deployed checkpoint was not exported separately. Figure 2 is plotted from the "
        "per-epoch train_loss / val_acc recorded by compare.py, and illustrates convergence differences under the matched "
        "ablation protocol.",
        east=ea,
    )

    add_heading_en(doc, "6.2 Quantitative Results (Test Set)", 2)
    add_para(doc, "Results from evaluate.py on 2,623 test images are as follows.", east=ea)
    add_para(doc, "Table 4  Overall test metrics", bold=True, center=True, size=10, east=ea)
    add_table(
        doc,
        ["Metric", "Value"],
        [
            ["Top-1 Accuracy", "97.03%"],
            ["Top-3 Accuracy", "99.24%"],
            ["Macro F1", "0.9686"],
            ["Weighted F1", "0.9703"],
        ],
        col_widths=[6, 4],
    )
    add_para(doc, "Table 5  Per-class precision / recall / F1 (test set)", bold=True, center=True, size=10, east=ea)
    add_table(doc, ["Class", "Precision", "Recall", "F1", "Support"], CLASS_ROWS, col_widths=[3, 2.5, 2.5, 2.5, 2.5])
    add_figure(doc, FIG / "confusion_matrix.png", "Figure 5  Confusion matrix on the test set")
    add_figure(doc, FIG / "per_class_f1.png", "Figure 6  Per-class F1 scores")

    add_heading_en(doc, "6.3 Ablation: Pretrained Fine-tuning vs. From-Scratch", 2)
    add_para(
        doc,
        "Using compare.py --epochs 10, we compare ImageNet-pretrained fine-tuning against training from scratch with "
        "identical data, seed, optimizer, learning rates, and epoch count. Metrics below are taken from "
        "output/compare_results.json.",
        east=ea,
    )
    add_para(doc, "Table 6  Pretrained vs from-scratch comparison", bold=True, center=True, size=10, east=ea)
    add_table(
        doc,
        ["Setting", "Best Val Accuracy", "Training Time"],
        [
            ["Pretrained (ImageNet init)", "96.99%", "2688.7 s (≈44.8 min)"],
            ["From-Scratch", "80.75%", "2388.6 s (≈39.8 min)"],
        ],
        col_widths=[5, 4.5, 3.5],
    )
    add_para(
        doc,
        "Interpretation: under the matched 10-epoch budget, pretrained fine-tuning clearly outperforms training from "
        "scratch (96.99% vs 80.75%), supporting effective ImageNet feature transfer to Animals-10. Absolute ablation "
        "numbers are lower than Section 6.2 because only 10 epochs are used, whereas the deployed model comes from full "
        "fine-tuning.",
        east=ea,
    )
    add_figure(doc, FIG / "compare_accuracy.png", "Figure 8  Pretrained vs from-scratch accuracy comparison")

    add_heading_en(doc, "6.4 Qualitative Analysis", 2)
    add_para(
        doc,
        "Beyond aggregate metrics, we inspect representative test samples (see images/demo_for_paper; summarized in Figure 7).",
        east=ea,
    )
    for t in [
        "Success cases: when the animal is salient and the background is moderately complex, Top-1 confidence is often near 100%, and Grad-CAM concentrates on the animal body/head.",
        "Competitive Top-3 cases: under stronger interference (e.g., sheep vs chicken), Top-1 may remain correct while Top-2 still holds about 10%–25% probability, showing the value of alternatives.",
        "Failure cases: the model may attend to the animal region yet predict the wrong class with high confidence. Grad-CAM explains spatial focus but does not guarantee label correctness.",
        "Low-confidence / OOD: non-animal images, heavy occlusion, or tiny objects can trigger the <50% warning.",
        "Multi-animal photos: only one global label is returned; detection-based methods are more suitable.",
    ]:
        p = doc.add_paragraph(t, style="List Bullet")
        for run in p.runs:
            set_run_font(run, east_asia=ea)
    add_figure(
        doc,
        FIG / "gradcam_examples.png",
        "Figure 7  Qualitative examples: success / Top-3 ambiguity / failure (left: input, right: Grad-CAM)",
        width_inches=5.6,
    )

    add_heading_en(doc, "6.5 Discussion", 2)
    add_para(
        doc,
        "Within an educational / demo-oriented recognition prototype, ResNet18 transfer learning is effective and "
        "interactive enough for desktop and web use. Section 6.2 shows strong test accuracy for the deployed model "
        "(Top-1 97.03%, Top-3 99.24%); Section 6.3 isolates ImageNet initialization under a matched protocol "
        "(pretrained 96.99% vs from-scratch 80.75%). "
        "Top-3, a confidence threshold, and Grad-CAM improve usability and interpretability, but high-confidence errors "
        "can still occur. Limitations include the closed ten-class taxonomy, imperfect softmax calibration, a short "
        "ablation budget, and no multi-object localization. Demo images under images/demo_for_paper can be used for "
        "screenshots; Figures 3–4 should be inserted after running window.py / myflask.py.",
        east=ea,
    )

    add_heading_en(doc, "7. Conclusion and Future Work", 1)
    add_para(
        doc,
        "This graduate group project delivered a complete animal image classification system based on ResNet18 and "
        "Animals-10, covering transfer learning, dual-interface deployment, offline evaluation, a pretrained/from-scratch "
        "ablation, and lightweight interpretability (Top-3, low-confidence warnings, Grad-CAM). On the test set, the fully "
        "fine-tuned system reaches 97.03% Top-1 and 99.24% Top-3 accuracy. Future work may explore multi-object detection, "
        "stronger open-set rejection, longer multi-seed ablations, and model compression.",
        east=ea,
    )

    add_heading_en(doc, "Acknowledgments", 1)
    add_para(doc, "The authors thank the course instructor and all group members for discussions and collaborative development.", east=ea)

    add_heading_en(doc, "References", 1)
    for r in FULL_REFS:
        add_para(doc, r, size=10, space_after=3, east=ea)

    add_heading_en(doc, "Appendix C  Group Contribution Statement", 1)
    add_table(
        doc,
        ["Member", "Role", "Main Contribution", "Estimated Effort (%)"],
        [
            ["[Name]", "Leader / algorithm", "[Fill in]", "[ ]"],
            ["[Name]", "System / GUI", "[Fill in]", "[ ]"],
            ["[Name]", "Web / experiments / writing", "[Fill in]", "[ ]"],
        ],
        col_widths=[3, 3.5, 5, 3.5],
    )

    doc.save(out_path)
    print("saved", out_path)


if __name__ == "__main__":
    build_zh(ROOT / "Course_Paper_Draft_zh.docx")
    build_en(ROOT / "Course_Paper_Draft.docx")
