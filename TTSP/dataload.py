"""
Dataset loaders for VStar Bench, HR-Bench, TreeBench, and MME-RealWorld-Lite.
"""
import os
import json
import base64
import tempfile
import ast
from io import BytesIO
from typing import List, Dict, Tuple, Optional

import pandas as pd
from PIL import Image
from datasets import load_dataset


# ============= VSTAR =============

def load_vstar_dataset(
    vstar_path: str, subset: str, max_questions: Optional[int] = None
) -> Tuple[List[str], str]:
    """Returns (sorted image file list, full subset path)."""
    test_path = os.path.join(vstar_path, subset)
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"VStar path not found: {test_path}")
    files = sorted(f for f in os.listdir(test_path) if not f.endswith('.json'))
    if max_questions:
        files = files[:max_questions]
    return files, test_path


def load_vstar_question(vstar_path: str, subset: str, img_file: str) -> Dict:
    img_path = os.path.abspath(os.path.join(vstar_path, subset, img_file))
    anno_path = os.path.abspath(os.path.join(vstar_path, subset, img_file.replace('.jpg', '.json')))
    if not os.path.exists(anno_path):
        raise FileNotFoundError(f"Annotation not found: {anno_path}")
    with open(anno_path) as f:
        anno = json.load(f)
    return {
        'question': anno['question'],
        'options': anno['options'],
        'img_path': img_path,
        'ground_truth': 'A',  # VStar always has A as correct answer
    }


# ============= HR-BENCH =============

def load_hrbench_dataset(
    hrbench_path: str,
    subset: str,
    category_filter: Optional[str] = None,
    max_questions: Optional[int] = None,
) -> Tuple[pd.DataFrame, str]:
    data_path = os.path.join(hrbench_path, subset + '.tsv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"HR-Bench file not found: {data_path}")
    df = pd.read_csv(data_path, sep='\t')
    if 'index' in df.columns:
        df = df.sort_values('index').reset_index(drop=True)
    if category_filter:
        df = df[df['category'].str.strip('"') == category_filter]
        print(f"HR-Bench category filter '{category_filter}': {len(df)} samples")
    if max_questions:
        df = df.iloc[:max_questions]
    return df, data_path


def load_hrbench_question(df: pd.DataFrame, qid: int) -> Dict:
    row = df.iloc[qid]
    options = [str(row['A']), str(row['B']), str(row['C']), str(row['D'])]
    image_data = base64.b64decode(row['image'])
    image = Image.open(BytesIO(image_data))
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    image.save(tmp.name)
    img_path = tmp.name
    tmp.close()
    return {
        'question': row['question'],
        'options': options,
        'img_path': img_path,
        'ground_truth': row['answer'],
    }


# ============= TREEBENCH =============

# TreeBench category mapping
TREEBENCH_CATEGORIES = {
    'Perception/Attributes': 'Perception/Attributes',
    'Perception/Material': 'Perception/Material',
    'Perception/Physical State': 'Perception/Physical State',
    'Perception/Object Retrieval': 'Perception/Object Retrieval',
    'Perception/OCR': 'Perception/OCR',
    'Reasoning/Perspective Transform': 'Reasoning/Perspective Transform',
    'Reasoning/Ordering': 'Reasoning/Ordering',
    'Reasoning/Contact and Occlusion': 'Reasoning/Contact and Occlusion',
    'Reasoning/Spatial Containment': 'Reasoning/Spatial Containment',
    'Reasoning/Comparison': 'Reasoning/Comparison',
}

def load_treebench_dataset(
    treebench_path: Optional[str] = None,
    category: Optional[str] = None,
    max_questions: Optional[int] = None,
) -> Tuple[pd.DataFrame, str]:
    """Load TreeBench dataset from local path or HuggingFace.
    
    Args:
        treebench_path: Local path to TreeBench.tsv file, if None load from HuggingFace
        category: Optional category filter from TREEBENCH_CATEGORIES
        max_questions: Maximum number of questions to load
        
    Returns:
        Tuple of (DataFrame, dataset identifier)
    """
    if treebench_path and os.path.exists(treebench_path):
        # Load from local file
        df = pd.read_csv(treebench_path, sep='\t')
    else:
        # Load from HuggingFace Hub
        df = load_dataset("HaochenWang/TreeBench", split="train").to_pandas()
    
    if category:
        df = df[df['category'] == category]
        print(f"TreeBench category filter '{category}': {len(df)} samples")
    
    if max_questions:
        df = df.iloc[:max_questions]
        
    return df, "treebench"


def load_treebench_question(df: pd.DataFrame, qid: int) -> Dict:
    """Load a single TreeBench question.
    
    Args:
        df: TreeBench DataFrame
        qid: Question index
        
    Returns:
        Dict with question, options, img_path, ground_truth, target_instances
    """
    row = df.iloc[qid]
    
    # Parse options from A, B, C, D columns (TreeBench format)
    # Columns A, B, C, D contain the option texts directly
    options = []
    for col in ['A', 'B', 'C', 'D']:
        if col in row and pd.notna(row[col]):
            options.append(str(row[col]).strip())
    
    # Decode base64 image
    image_data = base64.b64decode(row['image'])
    image = Image.open(BytesIO(image_data))
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    image.save(tmp.name)
    img_path = tmp.name
    tmp.close()
    
    # Parse target instances for IoU calculation
    target_instances = []
    try:
        target_instances = ast.literal_eval(row['target_instances'])
    except (ValueError, SyntaxError):
        pass
    
    return {
        'question': row['question'],
        'options': options,
        'img_path': img_path,
        'ground_truth': row['answer'].upper(),
        'category': row['category'],
        'target_instances': target_instances,
    }


# ============= MME-REALWORLD-LITE =============

def load_mme_realworld_lite_dataset(
    dataset_path: str,
    task_filter: Optional[str] = None,
    subtask_filter: Optional[str] = None,
    max_questions: Optional[int] = None,
) -> Tuple[List[Dict], str]:
    """Load MME-RealWorld-Lite dataset from JSON file.
    
    Args:
        dataset_path: Path to MME-RealWorld-Lite directory
        task_filter: Optional Task filter (e.g., 'Perception', 'Reasoning')
        subtask_filter: Optional Subtask filter (e.g., 'Remote Sensing')
        max_questions: Maximum number of questions to load
        
    Returns:
        Tuple of (list of question dicts, data identifier)
    """
    json_path = os.path.join(dataset_path, 'data', 'MME-RealWorld-Lite.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"MME-RealWorld-Lite data not found: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter out empty entries
    data = [item for item in data if item and 'Question_id' in item]
    
    # Apply filters
    if task_filter:
        data = [item for item in data if item.get('Task') == task_filter]
        print(f"MME-RealWorld-Lite Task filter '{task_filter}': {len(data)} samples")
    
    if subtask_filter:
        data = [item for item in data if item.get('Subtask') == subtask_filter]
        print(f"MME-RealWorld-Lite Subtask filter '{subtask_filter}': {len(data)} samples")
    
    if max_questions:
        data = data[:max_questions]
    
    return data, "mme_realworld_lite"


def load_mme_realworld_lite_question(data: List[Dict], qid: int, dataset_path: str) -> Dict:
    """Load a single MME-RealWorld-Lite question.
    
    Args:
        data: List of question dicts
        qid: Question index
        dataset_path: Base path to dataset
        
    Returns:
        Dict with question, options, img_path, ground_truth, task, subtask
    """
    item = data[qid]
    
    # Parse answer choices - remove (A), (B), etc. prefix
    raw_options = item.get('Answer choices', [])
    options = []
    for opt in raw_options:
        # Remove prefix like "(A) ", "(B) ", etc.
        if opt.startswith('(') and ')' in opt:
            options.append(opt.split(')', 1)[1].strip())
        else:
            options.append(opt.strip())
    
    # Build image path
    img_name = item.get('Image', '')
    img_path = os.path.join(dataset_path, 'data', 'imgs', img_name)
    
    return {
        'question': item.get('Text', ''),
        'options': options,
        'img_path': img_path,
        'ground_truth': item.get('Ground truth', ''),
        'task': item.get('Task', ''),
        'subtask': item.get('Subtask', ''),
        'category': item.get('Category', ''),
        'question_id': item.get('Question_id', ''),
        'question_type': item.get('Question Type', ''),
    }


# ============= REGISTRY =============

# Maps CLI --subset → (actual_path, dataset_type, category_filter)
SUBSET_REGISTRY = {
    'Attr':     ('direct_attributes', 'vstar',   None),
    'Spatial':  ('relative_position', 'vstar',   None),
    'HR4K-FSP': ('hr_bench_4k',       'hrbench', 'single'),
    'HR4K-FCP': ('hr_bench_4k',       'hrbench', 'cross'),
    'HR8K-FSP': ('hr_bench_8k',       'hrbench', 'single'),
    'HR8K-FCP': ('hr_bench_8k',       'hrbench', 'cross'),
    # TreeBench categories
    'TB-Attr':      ('treebench', 'treebench', 'Perception/Attributes'),
    'TB-Material':  ('treebench', 'treebench', 'Perception/Material'),
    'TB-Physical':  ('treebench', 'treebench', 'Perception/Physical State'),
    'TB-Retrieval': ('treebench', 'treebench', 'Perception/Object Retrieval'),
    'TB-OCR':       ('treebench', 'treebench', 'Perception/OCR'),
    'TB-Perspective': ('treebench', 'treebench', 'Reasoning/Perspective Transform'),
    'TB-Ordering':  ('treebench', 'treebench', 'Reasoning/Ordering'),
    'TB-Contact':   ('treebench', 'treebench', 'Reasoning/Contact and Occlusion'),
    'TB-Containment': ('treebench', 'treebench', 'Reasoning/Spatial Containment'),
    'TB-Comparison': ('treebench', 'treebench', 'Reasoning/Comparison'),
    # MME-RealWorld-Lite subsets
    'MME-Perception': ('mme_realworld_lite', 'mme_realworld_lite', 'Perception'),
    'MME-Reasoning':  ('mme_realworld_lite', 'mme_realworld_lite', 'Reasoning'),
}


def get_subset_info(subset: str) -> Tuple[str, str, Optional[str]]:
    if subset not in SUBSET_REGISTRY:
        raise ValueError(f"Unknown subset '{subset}'. Choices: {list(SUBSET_REGISTRY)}")
    return SUBSET_REGISTRY[subset]


def load_dataset(dataset_type: str, args, actual_subset: str, category_filter: Optional[str]):
    if dataset_type == 'vstar':
        files, data_path = load_vstar_dataset(args.dataset_path, actual_subset, args.max_questions)
        print(f"Loaded VStar '{actual_subset}': {len(files)} samples")
        return files, data_path
    elif dataset_type == 'hrbench':
        df, data_path = load_hrbench_dataset(
            args.dataset_path, actual_subset, category_filter, args.max_questions
        )
        print(f"Loaded HR-Bench '{actual_subset}': {len(df)} samples")
        return df, data_path
    elif dataset_type == 'treebench':
        # Try to find TreeBench.tsv in dataset_path
        treebench_tsv_path = None
        if hasattr(args, 'dataset_path') and args.dataset_path:
            potential_path = os.path.join(args.dataset_path, 'TreeBench.tsv')
            if os.path.exists(potential_path):
                treebench_tsv_path = potential_path
        df, data_path = load_treebench_dataset(
            treebench_path=treebench_tsv_path,
            category=category_filter,
            max_questions=args.max_questions,
        )
        print(f"Loaded TreeBench '{actual_subset}': {len(df)} samples")
        return df, data_path
    elif dataset_type == 'mme_realworld_lite':
        data, data_path = load_mme_realworld_lite_dataset(
            dataset_path=args.dataset_path,
            task_filter=category_filter,
            max_questions=args.max_questions,
        )
        print(f"Loaded MME-RealWorld-Lite '{actual_subset}': {len(data)} samples")
        return data, data_path
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def load_sample(
    qid: int,
    dataset_type: str,
    data_source,
    actual_subset: str,
    dataset_path: str,
) -> Dict:
    """Load a single sample and return question/options/ground_truth/img_path."""
    if dataset_type == 'vstar':
        img_file = data_source[qid]
        return load_vstar_question(dataset_path, actual_subset, img_file)
    elif dataset_type == 'treebench':
        return load_treebench_question(data_source, qid)
    elif dataset_type == 'mme_realworld_lite':
        return load_mme_realworld_lite_question(data_source, qid, dataset_path)
    else:
        return load_hrbench_question(data_source, qid)


def cleanup_sample(dataset_type: str, img_path: Optional[str]):
    """Remove temporary image file for HR-Bench and TreeBench samples."""
    if dataset_type in ('hrbench', 'treebench') and img_path and os.path.exists(img_path):
        try:
            os.remove(img_path)
        except Exception:
            pass
