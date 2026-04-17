import json
import numpy as np
from sys import argv, exit
from scipy.sparse import csc_array, lil_matrix
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from abc import ABC, abstractmethod
import numpy.linalg as la
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from fem_local_Plate20_DOF5_07_09_2025 import (
    Klocal, Flocal, Blocalbending, Dmatbending, Blocalmembrane, Dmatmembrane
)

OUTPUT_DIR = "output_python_square"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _as_float_list(values: Any, key_name: str) -> List[float]:
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError(f"'{key_name}' must be a non-empty list of numeric values.")
    out: List[float] = []
    for value in values:
        if not isinstance(value, (int, float)):
            raise ValueError(f"'{key_name}' contains a non-numeric value: {value}")
        out.append(float(value))
    return out


def _build_load_cases_from_nested_loads(loads_section: Dict[str, Any]) -> List[Dict[str, Any]]:
    required = ["x1", "x2", "y1", "y2", "q"]
    missing = [key for key in required if key not in loads_section]
    if missing:
        raise ValueError(f"Loads section is missing required keys: {missing}")

    x1_values = _as_float_list(loads_section["x1"], "loads.x1")
    x2_values = _as_float_list(loads_section["x2"], "loads.x2")
    y1_values = _as_float_list(loads_section["y1"], "loads.y1")
    y2_values = _as_float_list(loads_section["y2"], "loads.y2")
    q_values = _as_float_list(loads_section["q"], "loads.q")

    lengths = {len(x1_values), len(x2_values), len(y1_values), len(y2_values), len(q_values)}
    if len(lengths) != 1:
        raise ValueError("loads.x1/x2/y1/y2/q must all have the same number of entries.")

    load_cases: List[Dict[str, Any]] = []
    for idx in range(len(x1_values)):
        x1 = x1_values[idx]
        x2 = x2_values[idx]
        y1 = y1_values[idx]
        y2 = y2_values[idx]
        q = q_values[idx]

        if x2 <= x1:
            raise ValueError(f"Invalid load case {idx}: x2 must be greater than x1.")
        if y2 <= y1:
            raise ValueError(f"Invalid load case {idx}: y2 must be greater than y1.")
        if q < 0:
            raise ValueError(f"Invalid load case {idx}: q must be non-negative.")

        load_cases.append(
            {
                "rectangle": [x1, x2, y1, y2],
                "q": q,
            }
        )

    return load_cases


def read_unified_agent_input(filename: str, ndofs_per_node: int):
    with open(filename, "r") as jsonFS:
        json_data = json.load(jsonFS)

    if not isinstance(json_data, dict):
        raise ValueError("Input JSON must be a dictionary.")

    required_sections = ["nodes", "slab", "subgrade", "loads"]
    if not all(key in json_data for key in required_sections):
        raise ValueError(
            "Input JSON is not in unified agent format (expected keys: nodes, slab, subgrade, loads)."
        )

    nodes = json_data["nodes"]
    slab = json_data["slab"]
    subgrade = json_data["subgrade"]
    loads = json_data["loads"]

    if not isinstance(nodes, dict) or not isinstance(slab, dict) or not isinstance(subgrade, dict):
        raise ValueError("nodes/slab/subgrade sections must be dictionaries.")

    xnodes = np.array(_as_float_list(nodes.get("x"), "nodes.x"), dtype=float)
    ynodes = np.array(_as_float_list(nodes.get("y"), "nodes.y"), dtype=float)

    model_params = {
        "Emod": float(slab["Emod"]),
        "nu": float(slab["nu"]),
        "t": float(slab["t"]),
        "Kx": float(subgrade["Kx"]),
        "Ky": float(subgrade["Ky"]),
        "Kz": float(subgrade["Kz"]),
    }

    load_cases = _build_load_cases_from_nested_loads(loads)
    return mesh_plate_enumx(xnodes, ynodes, ndofs_per_node), model_params, load_cases


def write_legacy_vtk_results(mesh, output_path: str, point_fields: Dict[str, np.ndarray], displacement_vectors: np.ndarray):
    num_nodes = mesh.getNumNodes()
    num_elements = mesh.getNumElements()

    points = np.zeros((num_nodes, 3), dtype=float)
    for node_idx in range(num_nodes):
        x_coord, y_coord = mesh.getCoordinatesOfNode(node_idx)
        points[node_idx, 0] = x_coord
        points[node_idx, 1] = y_coord

    cells = [mesh.getNodesOfElement(elem_idx).astype(int).tolist() for elem_idx in range(num_elements)]

    with open(output_path, "w") as vtk_file:
        vtk_file.write("# vtk DataFile Version 3.0\n")
        vtk_file.write("Pavement FEA results\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET UNSTRUCTURED_GRID\n")

        vtk_file.write(f"POINTS {num_nodes} float\n")
        for point in points:
            vtk_file.write(f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f}\n")

        vtk_file.write(f"CELLS {num_elements} {num_elements * 5}\n")
        for cell in cells:
            vtk_file.write(f"4 {cell[0]} {cell[1]} {cell[2]} {cell[3]}\n")

        vtk_file.write(f"CELL_TYPES {num_elements}\n")
        for _ in range(num_elements):
            vtk_file.write("9\n")

        vtk_file.write(f"POINT_DATA {num_nodes}\n")
        for name, values in point_fields.items():
            flat = np.asarray(values, dtype=float).reshape(-1)
            if flat.size != num_nodes:
                raise ValueError(f"Point field '{name}' must have exactly {num_nodes} values.")
            vtk_file.write(f"SCALARS {name} float 1\n")
            vtk_file.write("LOOKUP_TABLE default\n")
            for value in flat:
                vtk_file.write(f"{value:.10e}\n")

        vectors = np.asarray(displacement_vectors, dtype=float)
        if vectors.shape != (num_nodes, 3):
            raise ValueError(f"Displacement vectors must have shape ({num_nodes}, 3).")
        vtk_file.write("VECTORS displacement float\n")
        for vec in vectors:
            vtk_file.write(f"{vec[0]:.10e} {vec[1]:.10e} {vec[2]:.10e}\n")


class mesh_plate(ABC):
    def __init__(self, xnodes: np.array, ynodes: np.array, num_dofs_per_node: int = 5):
        self.xnodes = xnodes
        self.ynodes = ynodes
        self.num_dofs_per_node = num_dofs_per_node
    
    def getNumNodes(self) -> int:
        return self.xnodes.size * self.ynodes.size
    
    def getNumNodes_x(self) -> int:
        return self.xnodes.size
    
    def getNumNodes_y(self) -> int:
        return self.ynodes.size
    
    def getNumElements(self) -> int:
        return (self.xnodes.size - 1) * (self.ynodes.size - 1)
    
    def getNumElements_x(self) -> int:
        return self.xnodes.size - 1
    
    def getNumElements_y(self) -> int:
        return self.ynodes.size - 1
    
    def getNumDofs(self) -> int:
        return self.getNumNodes() * self.num_dofs_per_node
    
    def getNumNodesPerElement(self) -> int:
        return 4
    
    def getNumDofsPerElement(self) -> int:
        return self.getNumNodesPerElement() * self.num_dofs_per_node
    
    @abstractmethod
    def getCoordinatesOfNode(self, nodeNum: int) -> np.array:
        pass
    
    @abstractmethod
    def getDofsOfNode(self, nodeNum: int) -> np.array:
        pass
        
    @abstractmethod
    def getNodesOfElement(self, elementNum: int) -> np.array:
        pass
    
    @abstractmethod
    def getCoordinatesOfNodesOfElement(self, elementNum: int) -> np.array:
        pass
    
    @abstractmethod
    def getDofsOfElement(self, elementNum: int) -> np.array:
        pass
    
    @abstractmethod
    def getSizeOfElement(self, elementNum: int) -> np.array:
        pass


class mesh_plate_enumx(mesh_plate):
    def __init__(self, xnodes: np.array, ynodes: np.array, num_dofs_per_node: int = 5):
        super().__init__(xnodes, ynodes, num_dofs_per_node)
    
    def getCoordinatesOfNode(self, nodeNum: int) -> np.array:
        assert nodeNum <= self.getNumNodes() - 1, (
            f"Node number {nodeNum} exceeds the maximum number of nodes, "
            f"which is {self.getNumNodes()-1}"
        )
        numNodesBeforeThisNode = nodeNum
        numRowsBeforeThisNode = numNodesBeforeThisNode % self.getNumNodes_x()
        numColsBeforeThisNode = numNodesBeforeThisNode // self.getNumNodes_x()
        xCoordOfThisNode = self.xnodes[numRowsBeforeThisNode]
        yCoordOfThisNode = self.ynodes[numColsBeforeThisNode]
        return np.array([xCoordOfThisNode, yCoordOfThisNode])
    
    def getDofsOfNode(self, nodeNum: int) -> np.array:
        assert nodeNum <= self.getNumNodes() - 1, (
            f"Node number {nodeNum} exceeds the maximum number of nodes, "
            f"which is {self.getNumNodes()-1}"
        )
        numDofsBeforeThisNode = nodeNum * self.num_dofs_per_node
        dofsOfThisNode = numDofsBeforeThisNode + np.arange(self.num_dofs_per_node)
        return dofsOfThisNode

    def getBendingDofsOfNode(self, nodeNum: int) -> np.array:
        all_dofs = self.getDofsOfNode(nodeNum)
        return all_dofs[:3]
        
    def getMembraneDofsOfNode(self, nodeNum: int) -> np.array:
        all_dofs = self.getDofsOfNode(nodeNum)
        return all_dofs[3:]

    def getBendingDofsOfElement(self, elementNum: int) -> np.array:
        nodesOfElement = self.getNodesOfElement(elementNum)
        bending_dofs = np.zeros((12,), dtype=int)
        for i, node in enumerate(nodesOfElement):
            bending_dofs[3 * i: 3 * (i + 1)] = self.getBendingDofsOfNode(node)
        return bending_dofs

    def getMembraneDofsOfElement(self, elementNum: int) -> np.array:
        nodesOfElement = self.getNodesOfElement(elementNum)
        membrane_dofs = np.zeros((8,), dtype=int)
        for i, node in enumerate(nodesOfElement):
            membrane_dofs[2 * i: 2 * (i + 1)] = self.getMembraneDofsOfNode(node)
        return membrane_dofs
    
    def getNodesOfElement(self, elementNum: int) -> np.array:
        assert elementNum <= self.getNumElements() - 1, (
            f"Element number {elementNum} exceeds the maximum number of elements, "
            f"which is {self.getNumElements()-1}"
        )
        numNodesFromRowsBefore = (elementNum // self.getNumElements_x()) * (self.getNumElements_x() + 1)
        numNodesBeforeThisElementInTheColumn = (elementNum % self.getNumElements_x())
        numNodesBeforeThisElement = numNodesFromRowsBefore + numNodesBeforeThisElementInTheColumn
        bottomLeftNodeOfThisElementInLocalOrder = numNodesBeforeThisElement
        topLeftNodeOfThisElementInLocalOrder = bottomLeftNodeOfThisElementInLocalOrder + 1
        topRightNodesOfThisElementInLocalOrder = topLeftNodeOfThisElementInLocalOrder + (self.getNumElements_x() + 1)
        bottomRightNodesOfThisElementInLocalOrder = topRightNodesOfThisElementInLocalOrder - 1
        return np.array([
            bottomLeftNodeOfThisElementInLocalOrder,
            topLeftNodeOfThisElementInLocalOrder,
            topRightNodesOfThisElementInLocalOrder,
            bottomRightNodesOfThisElementInLocalOrder
        ])
    
    def getCoordinatesOfNodesOfElement(self, elementNum: int) -> np.array:
        nodesOfElement = self.getNodesOfElement(elementNum)
        coordinatesOfNodesOfThisElementInLocalOrder = np.zeros((self.getNumNodesPerElement(), 2), dtype=float)
        for i, node in enumerate(nodesOfElement):
            coordinatesOfNodesOfThisElementInLocalOrder[i, :] = self.getCoordinatesOfNode(node)
        return coordinatesOfNodesOfThisElementInLocalOrder
    
    def getElementBounds(self, elem_num):
        coords = self.getCoordinatesOfNodesOfElement(elem_num)
        xmin = np.min(coords[:, 0])
        xmax = np.max(coords[:, 0])
        ymin = np.min(coords[:, 1])
        ymax = np.max(coords[:, 1])
        return xmin, xmax, ymin, ymax

    def getDofsOfElement(self, elementNum: int) -> np.array:
        nodesOfElement = self.getNodesOfElement(elementNum)
        dofsOfThisElementInLocalOrder = np.zeros((self.getNumDofsPerElement(),), dtype=int)
        for i, node in enumerate(nodesOfElement):
            dofsOfThisElementInLocalOrder[
                self.num_dofs_per_node * i: self.num_dofs_per_node * (i + 1)
            ] = self.getDofsOfNode(node)
        return dofsOfThisElementInLocalOrder
        
    def getSizeOfElement(self, elementNum: int) -> np.array:
        coordinatesOfNodesOfThisElementInLocalOrder = self.getCoordinatesOfNodesOfElement(elementNum)
        a = np.max(coordinatesOfNodesOfThisElementInLocalOrder[:, 0]) - np.min(coordinatesOfNodesOfThisElementInLocalOrder[:, 0])
        b = np.max(coordinatesOfNodesOfThisElementInLocalOrder[:, 1]) - np.min(coordinatesOfNodesOfThisElementInLocalOrder[:, 1])
        return np.array([a, b])


class Load:
    def __init__(self, mesh):
        self.mesh = mesh
        self.ndpn = mesh.num_dofs_per_node

    def getElementBounds(self, elem_num):
        coords = self.mesh.getCoordinatesOfNodesOfElement(elem_num)
        xmin = np.min(coords[:, 0])
        xmax = np.max(coords[:, 0])
        ymin = np.min(coords[:, 1])
        ymax = np.max(coords[:, 1])
        return xmin, xmax, ymin, ymax

    def getNodesAndDOFSInRectangle(self, x_min, x_max, y_min, y_max):
        x_nodes, y_nodes = self.mesh.xnodes, self.mesh.ynodes
        x_start = np.where(x_nodes <= x_min)[0][-1]
        x_end = np.where(x_nodes >= x_max)[0][0]
        y_start = np.where(y_nodes <= y_min)[0][-1]
        y_end = np.where(y_nodes >= y_max)[0][0]
        node_indices, dofs = [], []
        for j in range(y_start, y_end + 1):
            for i in range(x_start, x_end + 1):
                node_id = j * len(x_nodes) + i
                node_indices.append(node_id)
                dofs.extend(self.mesh.getDofsOfNode(node_id).tolist())
        return node_indices, dofs

    def computeOverlapArea(self, elementBounds, rectangularLoadBounds):
        exmin, exmax, eymin, eymax = elementBounds
        rxmin, rxmax, rymin, rymax = rectangularLoadBounds
        dx = min(exmax, rxmax) - max(exmin, rxmin)
        dy = min(eymax, rymax) - max(eymin, rymin)
        return dx * dy if dx > 0 and dy > 0 else 0

    def partitionLoadToElements(self, load_rect, total_load):
        x_min, x_max, y_min, y_max = load_rect
        rectangularLoadBounds = (x_min, x_max, y_min, y_max)
        x_nodes, y_nodes = self.mesh.xnodes, self.mesh.ynodes

        x_lower_index = np.where(x_nodes <= x_min)[0][-1]
        x_upper_index = np.where(x_nodes >= x_max)[0][0]
        y_lower_index = np.where(y_nodes <= y_min)[0][-1]
        y_upper_index = np.where(y_nodes >= y_max)[0][0]

        num_elems_x = self.mesh.getNumElements_x()
        x_el_start, x_el_end = x_lower_index, min(x_upper_index, num_elems_x)
        y_el_start, y_el_end = y_lower_index, min(y_upper_index, self.mesh.getNumElements_y())
        
        intersecting_elements, overlap_areas = [], []
        for j in range(y_el_start, y_el_end):
            for i in range(x_el_start, x_el_end):
                elem_id = j * num_elems_x + i
                elementBounds = self.getElementBounds(elem_id)
                overlap_area = self.computeOverlapArea(elementBounds, rectangularLoadBounds)
                if overlap_area > 0:
                    intersecting_elements.append(elem_id)
                    overlap_areas.append(overlap_area)
        
        total_overlap = sum(overlap_areas)
        if total_overlap == 0:
            return [], [], [], []
        
        print(f"\nTotal Load: {total_load} N over area {total_overlap:.4f} mm²")
        print(f"Intersecting Elements: {intersecting_elements}")

        load_distribution = []
        for i, e in enumerate(intersecting_elements):
            share = overlap_areas[i] / total_overlap
            partial_load = total_load * share
            load_distribution.append((e, partial_load))
            print(f" - Element {e}: overlap = {overlap_areas[i]:.4f} → load = {partial_load:.2f} N")
        
        nodesInsideLoad, dofsInsideLoad = self.getNodesAndDOFSInRectangle(*load_rect)
        print(f"Nodes inside load: {nodesInsideLoad}")
        print(f"DOFs inside load: {dofsInsideLoad}")
        return intersecting_elements, load_distribution, nodesInsideLoad, dofsInsideLoad

    def visualizeLoadPartition(
        self,
        intersecting_elements,
        load_distribution,
        rectangle,
        title="Load Partitioning",
        output_filename="load_region.png",
    ):
        fig, ax = plt.subplots(figsize=(8, 8))

        for nodeNum in range(self.mesh.getNumNodes()):
            x, y = self.mesh.getCoordinatesOfNode(nodeNum)
            ax.plot(y, x, 'ko', markersize=3)
        
        for elemNum in range(self.mesh.getNumElements()):
            xmin, xmax, ymin, ymax = self.mesh.getElementBounds(elemNum)
            ax.add_patch(
                Rectangle(
                    (ymin, xmin), ymax - ymin, xmax - xmin,
                    fill=False, edgecolor='blue', linewidth=0.6
                )
            )
            ax.text(
                0.5 * (ymin + ymax), 0.5 * (xmin + xmax), str(elemNum),
                color='black', ha='center', va='center',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.4)
            )
        
        num_colors = max(1, len(load_distribution))
        base_cmap = plt.colormaps.get_cmap('tab20')
        colors = [base_cmap(i / num_colors) for i in range(num_colors)]
        rxmin, rxmax, rymin, rymax = rectangle

        for elemNum in intersecting_elements:
            xmin, xmax, ymin, ymax = self.mesh.getElementBounds(elemNum)
            ax.add_patch(
                Rectangle(
                    (ymin, xmin), ymax - ymin, xmax - xmin,
                    fill=True, facecolor='none', hatch='///',
                    edgecolor='black', linewidth=1.5
                )
            )
        
        for idx, (elemNum, partial_load) in enumerate(load_distribution):
            xmin, xmax, ymin, ymax = self.mesh.getElementBounds(elemNum)
            overlap_xmin = max(xmin, rxmin)
            overlap_xmax = min(xmax, rxmax)
            overlap_ymin = max(ymin, rymin)
            overlap_ymax = min(ymax, rymax)

            if overlap_xmin < overlap_xmax and overlap_ymin < overlap_ymax:
                ov_width = overlap_ymax - overlap_ymin
                ov_height = overlap_xmax - overlap_xmin
                ax.add_patch(
                    Rectangle(
                        (overlap_ymin, overlap_xmin), ov_width, ov_height,
                        color=colors[idx], alpha=0.9,
                        label=f"Elem {elemNum}: {partial_load:.2f} N"
                    )
                )
        
        ax.add_patch(
            Rectangle(
                (rymin, rxmin), rymax - rymin, rxmax - rxmin,
                color='lightblue', alpha=0.5, label="Full load area"
            )
        )
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc='best')

        ax.set_xlabel('y (mm)')
        ax.set_ylabel('x (mm)')
        ax.set_title(title)
        ax.set_aspect('equal')
        plt.savefig(os.path.join(OUTPUT_DIR, output_filename), dpi=300)
        plt.close()


def plot_sparsity_pattern(matrix, title="Global Stiffness Matrix Sparsity Pattern"):
    plt.figure(figsize=(10, 10))
    plt.spy(matrix, markersize=1)
    plt.title(title)
    plt.xlabel('Columns (DOFs)')
    plt.ylabel('Rows (DOFs)')
    plt.savefig(os.path.join(OUTPUT_DIR, "global_stiffness_matrix_sparsity.png"), dpi=300)
    plt.close()




def write_summary_file(mesh, load_rectangles, total_load, w, sxx_top, sxx_bot, output_folder):
    summary_path = os.path.join(output_folder, "summary_results.txt")

    num_elements = mesh.getNumElements()
    if isinstance(load_rectangles, list) and len(load_rectangles) > 1:
        load_type = f"Rectangular ({len(load_rectangles)} cases)"
        load_area_mm2 = sum(
            max(0.0, (rect[1] - rect[0]) * (rect[3] - rect[2])) for rect in load_rectangles
        )
    else:
        load_type = "Rectangular"
        rect = load_rectangles[0] if isinstance(load_rectangles, list) and load_rectangles else load_rectangles
        load_area_mm2 = max(0.0, (rect[1] - rect[0]) * (rect[3] - rect[2]))

    max_disp_value = np.max(w)
    max_disp_abs = np.max(np.abs(w))
    max_disp_index = np.unravel_index(np.argmax(np.abs(w)), w.shape)
    max_disp_x = mesh.xnodes[max_disp_index[1]]
    max_disp_y = mesh.ynodes[max_disp_index[0]]

    max_sxx_top_value = np.max(sxx_top)
    max_sxx_top_abs = np.max(np.abs(sxx_top))
    max_sxx_top_index = np.unravel_index(np.argmax(np.abs(sxx_top)), sxx_top.shape)
    max_sxx_top_x = mesh.xnodes[max_sxx_top_index[1]]
    max_sxx_top_y = mesh.ynodes[max_sxx_top_index[0]]

    max_sxx_bot_value = np.max(sxx_bot)
    max_sxx_bot_abs = np.max(np.abs(sxx_bot))
    max_sxx_bot_index = np.unravel_index(np.argmax(np.abs(sxx_bot)), sxx_bot.shape)
    max_sxx_bot_x = mesh.xnodes[max_sxx_bot_index[1]]
    max_sxx_bot_y = mesh.ynodes[max_sxx_bot_index[0]]

    with open(summary_path, "w") as f:
        f.write("SUMMARY OF RESULTS\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Number of elements           : {num_elements}\n")
        f.write(f"Type of loading              : {load_type}\n")
        f.write(f"Total load (N)               : {total_load:.6f}\n")
        f.write(f"Load area (mm^2)             : {load_area_mm2:.6f}\n\n")

        f.write("MAXIMUM DISPLACEMENT\n")
        f.write("-" * 60 + "\n")
        f.write(f"Max displacement value (w)   : {max_disp_value:.6f} mm\n")
        f.write(f"Max |displacement|           : {max_disp_abs:.6f} mm\n")
        f.write(f"Location (x, y)              : ({max_disp_x}, {max_disp_y}) mm\n\n")

        f.write("MAXIMUM SIGMA XX AT TOP LAYER\n")
        f.write("-" * 60 + "\n")
        f.write(f"Max sigma_xx top             : {max_sxx_top_value:.6f} MPa\n")
        f.write(f"Max |sigma_xx top|           : {max_sxx_top_abs:.6f} MPa\n")
        f.write(f"Location (x, y)              : ({max_sxx_top_x}, {max_sxx_top_y}) mm\n\n")

        f.write("MAXIMUM SIGMA XX AT BOTTOM LAYER\n")
        f.write("-" * 60 + "\n")
        f.write(f"Max sigma_xx bottom          : {max_sxx_bot_value:.6f} MPa\n")
        f.write(f"Max |sigma_xx bottom|        : {max_sxx_bot_abs:.6f} MPa\n")
        f.write(f"Location (x, y)              : ({max_sxx_bot_x}, {max_sxx_bot_y}) mm\n")
def read_inputs(filename: str, ndofs_per_node: int) -> mesh_plate_enumx:
    try:
        with open(filename, 'r') as jsonFS:
            jsonData = json.load(jsonFS)
            xnodes = np.array(jsonData["nodes"]["x"], dtype=float)
            ynodes = np.array(jsonData["nodes"]["y"], dtype=float)
            return mesh_plate_enumx(xnodes, ynodes, ndofs_per_node)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit()


def read_element_parameters(filename: str):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error reading material/load input file: {e}")
        exit()


def read_load(filename: str):
    try:
        with open(filename, 'r') as jsonFS:
            jsonData = json.load(jsonFS)
            return jsonData["rectangle"], jsonData["magnitude"]
    except Exception as e:
        print(f"Error reading load file: {e}")
        exit()


if __name__ == "__main__":
    if len(argv) < 2:
        print("Usage: python DOF5_07.09.2025.py <agent_output_or_input_file.json>")
        exit()

    # Save all printed output to text file
    output_txt_path = os.path.join(OUTPUT_DIR, "results.txt")
    original_stdout = sys.stdout
    output_file = open(output_txt_path, "w")
    sys.stdout = output_file

    mesh = None
    load_case_data: List[Dict[str, Any]] = []
    total_applied_load = 0.0
    w = None
    sxx_top = None
    sxx_bot = None
    vtk_output_path = os.path.join(OUTPUT_DIR, "results.vtk")

    try:
        filename = argv[1]
        dofs_per_node = 5

        try:
            mesh, model_params, load_cases = read_unified_agent_input(filename, dofs_per_node)
            print(f"Loaded unified agent JSON input: {filename}")
        except Exception as unified_error:
            print(
                "Unified input parsing failed. Falling back to legacy static files. "
                f"Reason: {unified_error}"
            )
            mesh = read_inputs("mesh_inputDOF507-09-2025.json", dofs_per_node)
            legacy_params = read_element_parameters("inputstoDOF5_07_09_2025.json")
            legacy_load_rect, legacy_total_load = read_load("load_inputDOF507-09-2025.json")
            legacy_area = (legacy_load_rect[1] - legacy_load_rect[0]) * (legacy_load_rect[3] - legacy_load_rect[2])
            legacy_q = legacy_total_load / legacy_area if legacy_area > 0 else 0.0
            model_params = {
                "Emod": legacy_params["Emod"],
                "nu": legacy_params["nu"],
                "t": legacy_params["t"],
                "Kx": legacy_params["Kx"],
                "Ky": legacy_params["Ky"],
                "Kz": legacy_params["Kz"],
            }
            load_cases = [{"rectangle": legacy_load_rect, "q": legacy_q}]

        ndofs = mesh.getNumDofs()

        print("**** Testing the mesh class")
        for ielem in range(mesh.getNumElements()):
            print("Element DOFs: ", ielem, mesh.getDofsOfElement(ielem))
            print("Element size:", ielem, mesh.getSizeOfElement(ielem))
            print("Element node coords: ", ielem, mesh.getCoordinatesOfNodesOfElement(ielem))
            for node in mesh.getNodesOfElement(ielem):
                print("Node and coords: ", node, mesh.getCoordinatesOfNode(node))
                print("Node and DOFs: ", node, mesh.getDofsOfNode(node))
            print("****")

        print("**** Assembling Global Stiffness Matrix (Kg)")
        Kg = lil_matrix((ndofs, ndofs), dtype=float)
        Emod = model_params["Emod"]
        nu = model_params["nu"]
        t = model_params["t"]
        Kx = model_params["Kx"]
        Ky = model_params["Ky"]
        Kz = model_params["Kz"]

        for ielem in range(mesh.getNumElements()):
            dofs = mesh.getDofsOfElement(ielem)
            a_elem, b_elem = mesh.getSizeOfElement(ielem)
            Ke = Klocal(Emod, nu, a_elem, b_elem, t, Kx, Ky, Kz)
            for i_loc, I in enumerate(dofs):
                for j_loc, J in enumerate(dofs):
                    Kg[I, J] += Ke[i_loc, j_loc]

        print("\n**** Assembling Global Force Vector (Fg) from Rectangular Load(s)")
        load = Load(mesh)
        prepared_load_cases: List[Dict[str, Any]] = []
        total_applied_load = 0.0
        for idx, case in enumerate(load_cases):
            load_rect = [float(value) for value in case["rectangle"]]
            q_load_MPa = float(case["q"])
            load_area_mm2 = (load_rect[1] - load_rect[0]) * (load_rect[3] - load_rect[2])
            total_load = q_load_MPa * load_area_mm2
            total_applied_load += total_load

            print(f"\nLoad Case {idx + 1}:")
            print(f" - Rectangle: {load_rect}")
            print(f" - Area: {load_area_mm2:.4f} mm²")
            print(f" - Pressure q: {q_load_MPa:.6g} MPa")
            print(f" - Equivalent total load: {total_load:.6f} N")

            intersecting_elements, load_distribution, _, _ = load.partitionLoadToElements(load_rect, total_load)
            load.visualizeLoadPartition(
                intersecting_elements,
                load_distribution,
                load_rect,
                title=f"Load Partitioning - Case {idx + 1}",
                output_filename=f"load_region_case_{idx + 1}.png",
            )

            prepared_load_cases.append(
                {
                    "rectangle": load_rect,
                    "q": q_load_MPa,
                    "total_load": total_load,
                    "intersecting_elements": intersecting_elements,
                }
            )

        if not prepared_load_cases:
            raise ValueError("No valid load cases were found. Cannot assemble force vector.")

        load_case_data = prepared_load_cases

        n_time_steps = 5
        U_total = np.zeros(ndofs)

        for step in range(n_time_steps):
            print(f"\nProcessing Time Step {step + 1}/{n_time_steps}...")
            Fg_step = np.zeros((ndofs,))

            for case in prepared_load_cases:
                load_rect = case["rectangle"]
                q_load_MPa = case["q"]
                for elem_id in case["intersecting_elements"]:
                    xmin_el, xmax_el, ymin_el, ymax_el = mesh.getElementBounds(elem_id)
                    a_elem = xmax_el - xmin_el
                    b_elem = ymax_el - ymin_el

                    overlap_xmin_local = max(xmin_el, load_rect[0]) - xmin_el
                    overlap_xmax_local = min(xmax_el, load_rect[1]) - xmin_el
                    overlap_ymin_local = max(ymin_el, load_rect[2]) - ymin_el
                    overlap_ymax_local = min(ymax_el, load_rect[3]) - ymin_el

                    Fe_part = Flocal(
                        a_elem,
                        b_elem,
                        overlap_xmin_local,
                        overlap_xmax_local,
                        overlap_ymin_local,
                        overlap_ymax_local,
                        q_load_MPa,
                    )
                    Fe_step = Fe_part / n_time_steps
                    dofs = mesh.getDofsOfElement(elem_id)

                    for i_loc, I in enumerate(dofs):
                        Fg_step[I] += Fe_step[i_loc]

            print("Force vector for this time step:")
            print(Fg_step)

            U_step = np.zeros(ndofs)
            bc_dofs = []
            free_dofs = np.setdiff1d(np.arange(ndofs), bc_dofs)

            Kg_csr = Kg.tocsr()
            Kg_reduced = Kg_csr[free_dofs, :][:, free_dofs]
            Fg_reduced = Fg_step[free_dofs]
            
            if Kg_reduced.size > 0:
                U_reduced = la.solve(Kg_reduced.toarray(), Fg_reduced)
                U_step[free_dofs] = U_reduced

                print(f"\n--- Intermediate values for Step {step + 1} ---")
                print("Current total displacement (before this step's increment):")
                print(U_total)
                print("\nIncremental displacement for this step:")
                print(U_step)
                
                U_total += U_step
                
                print("\nUpdated total displacement:")
                print(U_total)
                print(f"--- End of Step {step + 1} ---\n")

        print("\n**** Post-processing: Calculating Nodal Stresses and Strains ****")
        
        nodal_stresses_top_sum = defaultdict(lambda: np.zeros(3))
        nodal_stresses_bottom_sum = defaultdict(lambda: np.zeros(3))
        nodal_stresses_membrane_sum = defaultdict(lambda: np.zeros(3))
        nodal_counts = defaultdict(int)

        D_b = Dmatbending(Emod, nu)
        D_m = Dmatmembrane(Emod, nu)

        for ielem in range(mesh.getNumElements()):
            dofs_elem = mesh.getDofsOfElement(ielem)
            U_elem = U_total[dofs_elem]
            a_elem, b_elem = mesh.getSizeOfElement(ielem)
            
            U_elem_reshaped = U_elem.reshape(mesh.getNumNodesPerElement(), mesh.num_dofs_per_node)
            U_elem_bending = U_elem_reshaped[:, :3].flatten()
            U_elem_membrane = U_elem_reshaped[:, 3:].flatten()
            
            nodes_elem = mesh.getNodesOfElement(ielem)
            coords_elem = mesh.getCoordinatesOfNodesOfElement(ielem)
            xmin_el = np.min(coords_elem[:, 0])
            ymin_el = np.min(coords_elem[:, 1])

            for i_loc, I_node in enumerate(nodes_elem):
                x_glob, y_glob = coords_elem[i_loc]
                x_loc = x_glob - xmin_el
                y_loc = y_glob - ymin_el
                
                B_b = Blocalbending(x_loc, y_loc, a_elem, b_elem)
                curvatures = B_b @ U_elem_bending
                strains_top = (t / 2.0) * curvatures
                strains_bottom = -(t / 2.0) * curvatures
                stresses_top = D_b @ strains_top
                stresses_bottom = D_b @ strains_bottom
                
                B_m = Blocalmembrane(x_loc, y_loc, a_elem, b_elem)
                strains_membrane = B_m @ U_elem_membrane
                stresses_membrane = D_m @ strains_membrane

                nodal_stresses_top_sum[I_node] += stresses_top
                nodal_stresses_bottom_sum[I_node] += stresses_bottom
                nodal_stresses_membrane_sum[I_node] += stresses_membrane
                nodal_counts[I_node] += 1

        num_nodes = mesh.getNumNodes()
        final_stresses_top = np.zeros((num_nodes, 3))
        final_stresses_bottom = np.zeros((num_nodes, 3))
        final_stresses_membrane = np.zeros((num_nodes, 3))

        for i in range(num_nodes):
            if nodal_counts[i] > 0:
                final_stresses_top[i, :] = nodal_stresses_top_sum[i] / nodal_counts[i]
                final_stresses_bottom[i, :] = nodal_stresses_bottom_sum[i] / nodal_counts[i]
                final_stresses_membrane[i, :] = nodal_stresses_membrane_sum[i] / nodal_counts[i]

        print("\n\n" + "=" * 28 + " FINAL RESULTS (5-DOF) " + "=" * 28)

        num_nodes_y = mesh.getNumNodes_y()
        num_nodes_x = mesh.getNumNodes_x()

        w = U_total[0::5].reshape((num_nodes_y, num_nodes_x))
        theta_x = U_total[1::5].reshape((num_nodes_y, num_nodes_x))
        theta_y = U_total[2::5].reshape((num_nodes_y, num_nodes_x))
        u = U_total[3::5].reshape((num_nodes_y, num_nodes_x))
        v = U_total[4::5].reshape((num_nodes_y, num_nodes_x))

        sxx_top = final_stresses_top[:, 0].reshape((num_nodes_y, num_nodes_x))
        syy_top = final_stresses_top[:, 1].reshape((num_nodes_y, num_nodes_x))
        sxy_top = -final_stresses_top[:, 2].reshape((num_nodes_y, num_nodes_x))

        sxx_bot = final_stresses_bottom[:, 0].reshape((num_nodes_y, num_nodes_x))
        syy_bot = final_stresses_bottom[:, 1].reshape((num_nodes_y, num_nodes_x))
        sxy_bot = -final_stresses_bottom[:, 2].reshape((num_nodes_y, num_nodes_x))

        sxx_mem = final_stresses_membrane[:, 0].reshape((num_nodes_y, num_nodes_x))
        syy_mem = final_stresses_membrane[:, 1].reshape((num_nodes_y, num_nodes_x))
        sxy_mem = final_stresses_membrane[:, 2].reshape((num_nodes_y, num_nodes_x))

        np.set_printoptions(precision=4, suppress=True)

        print("\n--- Displacements (w) in mm ---")
        print(w)

        print("\n--- Rotations about Y-axis (theta_x) in radians ---")
        print(theta_x)

        print("\n--- Rotations about X-axis (theta_y) in radians ---")
        print(theta_y)

        print("\n--- In-plane Displacements (u) in mm ---")
        print(u)

        print("\n--- In-plane Displacements (v) in mm ---")
        print(v)

        print("\n\n" + "=" * 20 + " BENDING STRESSES (MPa) " + "=" * 20)
        print("\n--- Stresses at Top Surface ---")
        print("\nSigma XX (top):")
        print(sxx_top)
        print("\nSigma YY (top):")
        print(syy_top)
        print("\nTau XY (top):")
        print(sxy_top)

        print("\n\n--- Stresses at Bottom Surface ---")
        print("\nSigma XX (bottom):")
        print(sxx_bot)
        print("\nSigma YY (bottom):")
        print(syy_bot)
        print("\nTau XY (bottom):")
        print(sxy_bot)

        print("\n\n" + "=" * 20 + " MEMBRANE STRESSES (MPa) " + "=" * 19)
        print("\n--- Stresses at Mid-plane ---")
        print("\nSigma XX (membrane):")
        print(sxx_mem)
        print("\nSigma YY (membrane):")
        print(syy_mem)
        print("\nTau XY (membrane):")
        print(sxy_mem)

        print("\n" + "=" * 75 + "\n")
        
        print("\n**** Final Global Force Vector (Fg) (Sum of all time steps) ****")
        Fg_final = np.zeros((ndofs,))
        for case in prepared_load_cases:
            load_rect = case["rectangle"]
            q_load_MPa = case["q"]
            for elem_id in case["intersecting_elements"]:
                xmin_el, xmax_el, ymin_el, ymax_el = mesh.getElementBounds(elem_id)
                a_elem = xmax_el - xmin_el
                b_elem = ymax_el - ymin_el

                overlap_xmin_local = max(xmin_el, load_rect[0]) - xmin_el
                overlap_xmax_local = min(xmax_el, load_rect[1]) - xmin_el
                overlap_ymin_local = max(ymin_el, load_rect[2]) - ymin_el
                overlap_ymax_local = min(ymax_el, load_rect[3]) - ymin_el

                Fe_part = Flocal(
                    a_elem,
                    b_elem,
                    overlap_xmin_local,
                    overlap_xmax_local,
                    overlap_ymin_local,
                    overlap_ymax_local,
                    q_load_MPa,
                )
                dofs = mesh.getDofsOfElement(elem_id)

                for i_loc, I in enumerate(dofs):
                    Fg_final[I] += Fe_part[i_loc]
        
        print("Fg_final shape:", Fg_final.shape)
        print(Fg_final)

        print("\n**** Global Stiffness Matrix (Kg) ****")
        print("Kg shape:", Kg.shape)
        if Kg.size < 500:
            print(Kg.toarray())
        else:
            print("Matrix is too large to display. Printing a snippet instead:")
            print(Kg[:20, :20].toarray())
        
        print("\n**** Plotting Global Stiffness Matrix Sparsity Pattern ****")
        plot_sparsity_pattern(Kg)

        print("\n**** Solving the System for Displacement (U)")
        print("\nFull displacement vector U (sum of all steps):")
        print(U_total)
        print(f"\nTotal DOFs: {ndofs}, Free DOFs: {len(free_dofs)}, Constrained DOFs: {len(bc_dofs)}")

        displacement_vectors = np.column_stack((u.reshape(-1), v.reshape(-1), w.reshape(-1)))
        point_fields = {
            "w": w,
            "theta_x": theta_x,
            "theta_y": theta_y,
            "u": u,
            "v": v,
            "sxx_top": sxx_top,
            "syy_top": syy_top,
            "sxy_top": sxy_top,
            "sxx_bottom": sxx_bot,
            "syy_bottom": syy_bot,
            "sxy_bottom": sxy_bot,
            "sxx_membrane": sxx_mem,
            "syy_membrane": syy_mem,
            "sxy_membrane": sxy_mem,
        }
        write_legacy_vtk_results(mesh, vtk_output_path, point_fields, displacement_vectors)
        print(f"VTK results saved at: {vtk_output_path}")

    finally:
        sys.stdout = original_stdout
        output_file.close()

        if mesh is not None and w is not None and sxx_top is not None and sxx_bot is not None:
            write_summary_file(
                mesh=mesh,
                load_rectangles=[case["rectangle"] for case in load_case_data],
                total_load=total_applied_load,
                w=w,
                sxx_top=sxx_top,
                sxx_bot=sxx_bot,
                output_folder=OUTPUT_DIR,
            )

        print(f"Results saved in: {output_txt_path}")
        print(f"Summary saved in: {os.path.join(OUTPUT_DIR, 'summary_results.txt')}")
        if os.path.exists(vtk_output_path):
            print(f"VTK saved in: {vtk_output_path}")