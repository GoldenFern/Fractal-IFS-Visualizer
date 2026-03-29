"""
分形几何可视化：IFS系统。
目前先在二维平面上进行操作。
可视化方法：
1. 用户构造IFS系统的压缩映射函数；
2. 确定分形所在的集合的范围；
3. 确定初始点的选择（默认随机选择一个点）；
4. 通过构造迭代序列，剔除前m个点，总共n个点，作为分形的近似绘制。

绘制方法的选择：
1. 基于matplotlib绘制
2. 基于plotly绘制
可选是否保存。matplotlib保存为jpg格式（清晰度尽可能高），plotly保存为html格式
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import json
from typing import List, Tuple, Dict, Any, Callable

class IFSVisualizer:
    """
    Iterated Function System (IFS) Visualizer for 2D Fractal Geometry.

    Attributes:
        functions (list): A list of callable functions representing contraction mappings.
                          Each function should take (x, y) and return (new_x, new_y).
        probabilities (list): A list of probabilities corresponding to each function.
                              Must sum to 1.
        points (np.ndarray): Generated points (x, y) after iteration.
    """

    def __init__(self, functions: List[Callable[[float, float], Tuple[float, float]]], probabilities: List[float] = None):
        self.functions = functions
        # If no probabilities provided, assume uniform distribution
        if probabilities is None:
            n = len(functions)
            self.probabilities = [1.0 / n] * n
        else:
            self.probabilities = probabilities

        # Validate probabilities sum to 1 (approx)
        if not np.isclose(sum(self.probabilities), 1.0):
            raise ValueError("Probabilities must sum to 1.")

        self.points = None

    def generate_points(self, n_total: int, m_discard: int, initial_point: Tuple[float, float] = (0.0, 0.0)):
        """
        Generates fractal points using the Chaos Game algorithm.

        Args:
            n_total (int): Total number of iterations.
            m_discard (int): Number of initial points to discard (transient state).
            initial_point (tuple): Starting coordinate (x, y).
        """
        # Pre-allocate array for efficiency
        # We only store points that are NOT discarded to save memory
        n_keep = n_total - m_discard
        if n_keep <= 0:
            raise ValueError("n_total must be greater than m_discard")

        x, y = initial_point

        # To speed up, we pre-select the sequence of functions to apply
        # This avoids calling random.choice inside the loop
        func_indices = np.random.choice(
            len(self.functions),
            size=n_total,
            p=self.probabilities
        )

        # Temporary lists for collection (using lists is often faster than resizing numpy arrays in loops)
        x_list = []
        y_list = []

        # Iteration
        for i, idx in enumerate(func_indices):
            # Apply the selected function
            x, y = self.functions[idx](x, y)

            # Only append if we are past the discard threshold
            if i >= m_discard:
                x_list.append(x)
                y_list.append(y)

        self.points = np.column_stack((x_list, y_list))
        print(f"Generated {len(self.points)} points after discarding first {m_discard}.")

    def plot_matplotlib(self, save: bool = False, filename: str = "ifs_fractal.jpg", color: Any = 'green', alpha: float = 0.4):
        """
        Visualizes the fractal using Matplotlib (Static).
        """
        if self.points is None:
            print("No points generated. Please run generate_points() first.")
            return

        plt.figure(figsize=(15, 15))
        # fig = plt.gcf()
        # fig.patch.set_facecolor((190/255, 220/255, 230/255)) # 示例：调高RGB值
        # ax = plt.gca()
        # ax.set_facecolor((190/255, 220/255, 230/255))

        plt.scatter(self.points[:, 0], self.points[:, 1], s=0.2, color=color, alpha=alpha)


        plt.axis('equal')
        plt.xticks([])
        plt.yticks([])
        plt.box(False) # 隐藏边框
        plt.grid(False)  # Fractals often look better without grids

        if save:
            # Save with high DPI as requested
            plt.savefig(filename, dpi=700, bbox_inches='tight')
            print(f"Image saved to {os.path.abspath(filename)}")

        plt.show()

    def plot_plotly(self, save: bool = False, filename: str = "ifs_fractal.html", color: Any = 'green'):
        """
        Visualizes the fractal using Plotly (Interactive).
        Uses WebGL (Scattergl) for performance with large datasets.
        """
        if self.points is None:
            print("No points generated. Please run generate_points() first.")
            return

        fig = go.Figure(data=go.Scattergl(
            x=self.points[:, 0],
            y=self.points[:, 1],
            mode='markers',
            marker=dict(
                size=1.5,
                color=color,
                opacity=0.6
            ),
            name='Fractal Points'
        ))

        fig.update_layout(
            title=r"$\text{Interactive IFS Fractal}$",
            xaxis_title=r"$x$",
            yaxis_title=r"$y$",
            template="plotly_white",
            width=900,
            height=900,
            showlegend=False
        )

        # Fix aspect ratio
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1
        )

        if save:
            fig.write_html(filename)
            print(f"Interactive plot saved to {os.path.abspath(filename)}")

        fig.show()


# ==========================================
# Common Fractal IFS Definitions (保留原有功能)
# ==========================================

def define_barnsley_fern():
    """
    Defines the 4 affine transformations for the Barnsley Fern.
    Returns functions and their probabilities.
    """
    # Transformation 1: Stem (Probability 0.01)
    f1 = lambda x, y: (0.0, 0.16 * y)

    # Transformation 2: Successively smaller leaflets (Probability 0.85)
    f2 = lambda x, y: (0.85 * x + 0.04 * y, -0.04 * x + 0.85 * y + 1.6)

    # Transformation 3: Largest left-hand leaflet (Probability 0.07)
    f3 = lambda x, y: (0.2 * x - 0.26 * y, 0.23 * x + 0.22 * y + 1.6)

    # Transformation 4: Largest right-hand leaflet (Probability 0.07)
    f4 = lambda x, y: (-0.15 * x + 0.28 * y, 0.26 * x + 0.24 * y + 0.44)

    funcs = [f1, f2, f3, f4]
    probs = [0.01, 0.85, 0.07, 0.07]
    return funcs, probs


def define_sierpinski_triangle():
    """
    Defines the 3 affine transformations for the Sierpinski Triangle.
    """
    # Three contractions towards the vertices of an equilateral triangle
    f1 = lambda x, y: (0.5 * x, 0.5 * y)  # Bottom left
    f2 = lambda x, y: (0.5 * x + 0.5, 0.5 * y)  # Bottom right
    f3 = lambda x, y: (0.5 * x + 0.25, 0.5 * y + 0.5)  # Top

    funcs = [f1, f2, f3]
    probs = [1/3, 1/3, 1/3]  # Equal probabilities
    return funcs, probs


def define_koch_curve():
    """
    Defines the 4 affine transformations for the Koch Curve.
    """
    # Four transformations that create the Koch curve pattern
    f1 = lambda x, y: (x/3, y/3)  # First segment
    f2 = lambda x, y: (x/3 * np.cos(np.pi/3) - y/3 * np.sin(np.pi/3) + 1/3,
                       x/3 * np.sin(np.pi/3) + y/3 * np.cos(np.pi/3))  # Second segment with rotation
    f3 = lambda x, y: (x/3 * np.cos(-np.pi/3) - y/3 * np.sin(-np.pi/3) + 1/2,
                       x/3 * np.sin(-np.pi/3) + y/3 * np.cos(-np.pi/3) + np.sqrt(3)/6)  # Third segment
    f4 = lambda x, y: (x/3 + 2/3, y/3)  # Fourth segment

    funcs = [f1, f2, f3, f4]
    probs = [0.25, 0.25, 0.25, 0.25]  # Equal probabilities
    return funcs, probs


def define_cantor_dust():
    """
    Defines the 4 affine transformations for Cantor Dust.
    A 2D generalization of the Cantor set.
    """
    # Four transformations that remove the central square
    scale = 1/3
    f1 = lambda x, y: (scale * x, scale * y)  # Bottom left
    f2 = lambda x, y: (scale * x + 2*scale, scale * y)  # Bottom right
    f3 = lambda x, y: (scale * x, scale * y + 2*scale)  # Top left
    f4 = lambda x, y: (scale * x + 2*scale, scale * y + 2*scale)  # Top right

    funcs = [f1, f2, f3, f4]
    probs = [0.25, 0.25, 0.25, 0.25]  # Equal probabilities
    return funcs, probs


def define_levy_curve():
    """
    Defines the 2 affine transformations for the Lévy C Curve.
    """
    # Two transformations that create the Lévy C curve pattern
    sqrt2 = np.sqrt(2)/2
    f1 = lambda x, y: (0.5 * x - 0.5 * y, 0.5 * x + 0.5 * y)  # Rotation and scaling
    f2 = lambda x, y: (0.5 * x + 0.5 * y + 0.5, -0.5 * x + 0.5 * y + 0.5)  # Rotation, scaling, translation

    funcs = [f1, f2]
    probs = [0.5, 0.5]  # Equal probabilities
    return funcs, probs


def define_dragon_curve():
    """
    Defines the 2 affine transformations for the Dragon Curve.
    """
    # Two transformations that create the dragon curve pattern
    f1 = lambda x, y: (0.5 * x - 0.5 * y, 0.5 * x + 0.5 * y)  # First transformation
    f2 = lambda x, y: (-0.5 * x - 0.5 * y + 1, 0.5 * x - 0.5 * y)  # Second transformation

    funcs = [f1, f2]
    probs = [0.5, 0.5]  # Equal probabilities
    return funcs, probs


def define_ifs_tree():
    """
    Defines a simple tree-like fractal using IFS.
    """
    # Four transformations creating a branching pattern
    f1 = lambda x, y: (0.05 * x, 0.6 * y)  # Trunk
    f2 = lambda x, y: (0.45 * x - 0.45 * y, 0.45 * x + 0.45 * y + 0.6)  # Left branch
    f3 = lambda x, y: (0.45 * x + 0.45 * y, -0.45 * x + 0.45 * y + 0.6)  # Right branch
    f4 = lambda x, y: (0.5 * x, 0.5 * y + 0.5)  # Central growth

    funcs = [f1, f2, f3, f4]
    probs = [0.1, 0.3, 0.3, 0.3]  # Different probabilities for natural look
    return funcs, probs


def define_spiral_ifs():
    """
    Defines a spiral pattern using IFS.
    """
    # Creates a spiral pattern with rotation and scaling
    angle = np.pi / 6  # 30 degrees
    scale = 0.9
    f1 = lambda x, y: (scale * (x * np.cos(angle) - y * np.sin(angle)),
                       scale * (x * np.sin(angle) + y * np.cos(angle)))
    f2 = lambda x, y: (scale * (x * np.cos(-angle) - y * np.sin(-angle)) + 0.5,
                       scale * (x * np.sin(-angle) + y * np.cos(-angle)) + 0.5)

    funcs = [f1, f2]
    probs = [0.5, 0.5]  # Equal probabilities
    return funcs, probs


class RandomFractalGenerator:
    """
    随机分形函数生成器。
    生成随机的压缩映射和概率分布来创建分形。
    """

    def __init__(self):
        self.n: int = None
        self.matrices: List[List[List[float]]] = None
        self.translations: List[List[float]] = None
        self.probabilities: List[float] = None
        self.functions: List[Callable[[float, float], Tuple[float, float]]] = None

    def generate_random_fractal(self) -> Dict[str, Any]:
        """
        生成一个随机的分形系统。

        Returns:
            Dict containing all generated parameters and functions
        """
        # 1. 随机选择n (2~5之间)
        self.n = np.random.randint(2, 6) # 6 is exclusive, so 2, 3, 4, 5
        print(f"生成 {self.n} 个压缩映射")

        # 2. 生成n个随机的压缩映射矩阵和对应的平移向量
        self.matrices = []
        self.translations = []
        self.functions = []

        for i in range(self.n):
            # 生成随机矩阵，确保是压缩映射（谱范数 < 1）
            matrix = self._generate_contraction_matrix()
            translation = np.random.uniform(-1, 1, 2) # 平移向量在[-1, 1]之间

            self.matrices.append(matrix.tolist())
            self.translations.append(translation.tolist())

            # 创建对应的函数 (仿射变换: x' = A*x + b)
            # 使用闭包捕获当前的matrix和translation值
            func = lambda x, y, m=matrix, t=translation: (
                m[0, 0] * x + m[0, 1] * y + t[0],
                m[1, 0] * x + m[1, 1] * y + t[1]
            )
            self.functions.append(func)

        # 3. 生成概率数组，倾向于等概率分布
        self.probabilities = self._generate_probabilities()

        return {
            'n': self.n,
            'matrices': self.matrices,
            'translations': self.translations,
            'probabilities': self.probabilities,
            'functions': self.functions # 注意：函数本身不能直接序列化到JSON
        }

    def _generate_contraction_matrix(self) -> np.ndarray:
        """
        生成一个压缩映射矩阵（谱范数 < 1）。
        """
        max_attempts = 100
        for attempt in range(max_attempts):
            # 生成随机矩阵，元素在[-1, 1]之间
            matrix = np.random.uniform(-1, 1, (2, 2))

            # 计算谱范数（最大奇异值）
            spectral_norm = np.linalg.norm(matrix, 2)

            # 如果谱范数大于0，则进行缩放
            if spectral_norm > 0:
                # 随机选择一个压缩因子 (0.3 ~ 0.8)，确保矩阵是压缩的
                contraction_factor = np.random.uniform(0.3, 0.8)
                matrix = matrix / spectral_norm * contraction_factor

                # 再次检查谱范数，确保其小于1
                if np.linalg.norm(matrix, 2) < 1:
                    return matrix

        # 如果多次尝试失败，返回一个已知的、安全的压缩映射
        print("Warning: Failed to generate a random contraction matrix after many attempts. Using a default one.")
        return np.array([[0.5, 0], [0, 0.5]])

    def _generate_probabilities(self) -> List[float]:
        """
        生成概率数组，倾向于等概率分布。
        """
        # 生成接近均匀分布的随机权重，范围在[0.8, 1.2]之间
        weights = np.random.uniform(0.8, 1.2, self.n)

        # 归一化，使总和为1
        probabilities = weights / np.sum(weights)

        return probabilities.tolist()

    def save_parameters(self, filename: str):
        """
        保存生成的参数到JSON文件。
        """
        if self.n is None:
            raise ValueError("请先生成随机分形 (call generate_random_fractal()).")

        # functions对象不能直接序列化，所以不包含在保存的参数中
        parameters = {
            'n_mappings': self.n,
            'matrices': self.matrices,
            'translations': self.translations,
            'probabilities': self.probabilities,
            'generation_time': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(parameters, f, indent=2, ensure_ascii=False)

        print(f"参数已保存到 {filename}")


def generate_matplotlib_color():
    """
    随机生成不浅不深、适用于matplotlib绘图的颜色。
    """
    r = np.random.uniform(0.2, 0.8)
    g = np.random.uniform(0.2, 0.8)
    b = np.random.uniform(0.2, 0.8)
    return (r, g, b)

def plot_all_functions():
    function_list = [
        define_cantor_dust, define_levy_curve, define_dragon_curve, define_ifs_tree,
        define_sierpinski_triangle, define_barnsley_fern, define_spiral_ifs, define_koch_curve
    ]

    for func in function_list:
        ifs_funcs, ifs_probs = func()

        # 创建可视化器实例
        visualizer = IFSVisualizer(ifs_funcs, ifs_probs)

        # 生成点集
        visualizer.generate_points(
            n_total=300000,
            m_discard=3000,
            initial_point=(0.1, 0.1)
        )

        visualizer.plot_matplotlib(
            save=True,
            filename=f'{time.strftime("%Y%m%d_%H%M%S", time.localtime())}.png',
            color=generate_matplotlib_color(),
        )

def my_ifs():
    # Transformation 1: Stem
    f1 = lambda x, y: (0.0, 0.16 * y)

    # Transformation 2: Successively smaller leaflets
    f2 = lambda x, y: (0.85 * x + 0.08*np.sin(y), 0.85 * y + 1.6)

    # Transformation 3: Largest left-hand leaflet
    f3 = lambda x, y: (0.2 * x - 0.31 * y, 0.25 * x + 0.2 * y + 1.6)

    # Transformation 4: Largest right-hand leaflet
    f4 = lambda x, y: (-0.15 * x + 0.33 * y, 0.28 * x + 0.22 * y + 0.44)

    funcs = [f1, f2, f3, f4]
    probs = [0.02, 0.84, 0.07, 0.07]
    return funcs, probs


if __name__ == "__main__":
    visualizer = IFSVisualizer(*my_ifs())
    visualizer.generate_points(1000000, 2000, initial_point=(0.1, 0.1))
    visualizer.plot_matplotlib(save=True, color="gold", alpha=0.2, filename="IFS.png")