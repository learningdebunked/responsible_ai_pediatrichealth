# Design Document: Performance Analysis and Visualization System

## Overview

The Performance Analysis and Visualization System is a Python-based tool that ingests model performance data across multiple developmental delay types and generates comprehensive analytical visualizations. The system processes tabular performance metrics (AUROC, Precision, Recall, F1, Lead Time) from various machine learning models (LogReg, RF, GRU-FL, Trans-FL) and produces publication-quality charts, statistical analyses, and summary reports.

The system is designed to be modular, extensible, and easy to integrate into existing research workflows. It will be implemented as a standalone Python script that can be invoked from the command line or imported as a module.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Layer                               │
│         (CSV, JSON, Python Dict, DataFrame)                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                 Validation Layer                             │
│      (Schema validation, Range checks, Type checks)         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Analysis Layer                              │
│   (Statistical tests, Aggregations, Comparisons)           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│               Visualization Layer                            │
│  (Matplotlib/Seaborn charts, Multi-panel layouts)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Export Layer                                │
│        (PNG/PDF/SVG images, Markdown reports, CSV)         │
└─────────────────────────────────────────────────────────────┘
```

### Component Architecture

The system consists of five main components:

1. **DataLoader**: Handles input parsing and validation
2. **PerformanceAnalyzer**: Computes statistics and comparisons
3. **VisualizationEngine**: Generates charts and plots
4. **ReportGenerator**: Creates summary reports
5. **ExportManager**: Handles file output in multiple formats

## Components and Interfaces

### 1. DataLoader

**Responsibility**: Load and validate performance data from various sources.

**Interface**:
```python
class DataLoader:
    def load_from_csv(self, filepath: str) -> pd.DataFrame
    def load_from_json(self, filepath: str) -> pd.DataFrame
    def load_from_dict(self, data: dict) -> pd.DataFrame
    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]
    def validate_ranges(self, df: pd.DataFrame) -> Tuple[bool, List[str]]
```

**Key Methods**:
- `load_from_csv()`: Parse CSV file into DataFrame
- `load_from_json()`: Parse JSON file into DataFrame
- `load_from_dict()`: Convert Python dictionary to DataFrame
- `validate_schema()`: Check for required columns
- `validate_ranges()`: Verify metric values are within valid bounds

### 2. PerformanceAnalyzer

**Responsibility**: Compute statistical analyses and performance comparisons.

**Interface**:
```python
class PerformanceAnalyzer:
    def __init__(self, data: pd.DataFrame)
    def compute_summary_statistics(self) -> pd.DataFrame
    def compare_models(self, metric: str) -> pd.DataFrame
    def compare_delay_types(self, metric: str) -> pd.DataFrame
    def compute_fl_performance_gap(self) -> Dict[str, float]
    def identify_best_models(self) -> Dict[str, str]
    def compute_effect_sizes(self, model1: str, model2: str) -> Dict[str, float]
    def compute_confidence_intervals(self, confidence: float = 0.95) -> pd.DataFrame
```

**Key Methods**:
- `compute_summary_statistics()`: Calculate mean, std, min, max for all metrics
- `compare_models()`: Generate pairwise model comparisons for a specific metric
- `compare_delay_types()`: Analyze performance patterns across delay types
- `compute_fl_performance_gap()`: Calculate performance difference between FL and non-FL models
- `identify_best_models()`: Find top-performing model for each delay type
- `compute_effect_sizes()`: Calculate Cohen's d for model differences
- `compute_confidence_intervals()`: Generate confidence intervals for metrics

### 3. VisualizationEngine

**Responsibility**: Generate publication-quality visualizations.

**Interface**:
```python
class VisualizationEngine:
    def __init__(self, data: pd.DataFrame, analyzer: PerformanceAnalyzer)
    def plot_model_comparison(self, delay_type: str, output_path: str)
    def plot_delay_type_heatmap(self, metric: str, output_path: str)
    def plot_fl_comparison(self, output_path: str)
    def plot_lead_time_analysis(self, output_path: str)
    def plot_comprehensive_dashboard(self, output_path: str)
    def plot_metric_distribution(self, metric: str, output_path: str)
    def set_style(self, style: str = 'publication')
```

**Key Methods**:
- `plot_model_comparison()`: Grouped bar chart comparing models for a delay type
- `plot_delay_type_heatmap()`: Heatmap showing metric values across delay types and models
- `plot_fl_comparison()`: Side-by-side comparison of FL vs non-FL models
- `plot_lead_time_analysis()`: Box plots and bar charts for lead time metrics
- `plot_comprehensive_dashboard()`: Multi-panel dashboard with all key visualizations
- `plot_metric_distribution()`: Distribution plots for specific metrics
- `set_style()`: Configure matplotlib style for publication quality

### 4. ReportGenerator

**Responsibility**: Create textual summary reports.

**Interface**:
```python
class ReportGenerator:
    def __init__(self, data: pd.DataFrame, analyzer: PerformanceAnalyzer)
    def generate_markdown_report(self, output_path: str)
    def generate_summary_statistics_csv(self, output_path: str)
    def generate_findings_text(self) -> str
    def generate_model_ranking(self) -> str
```

**Key Methods**:
- `generate_markdown_report()`: Create comprehensive markdown report with tables and findings
- `generate_summary_statistics_csv()`: Export computed statistics to CSV
- `generate_findings_text()`: Generate natural language summary of key findings
- `generate_model_ranking()`: Create ranked list of models by performance

### 5. ExportManager

**Responsibility**: Handle file exports in multiple formats.

**Interface**:
```python
class ExportManager:
    def __init__(self, output_dir: str)
    def save_figure(self, fig: plt.Figure, name: str, formats: List[str] = ['png', 'pdf', 'svg'])
    def save_dataframe(self, df: pd.DataFrame, name: str, format: str = 'csv')
    def save_text(self, content: str, name: str, format: str = 'md')
    def create_output_directory(self)
    def generate_filename(self, base_name: str, extension: str) -> str
```

**Key Methods**:
- `save_figure()`: Save matplotlib figure in multiple formats with high DPI
- `save_dataframe()`: Export DataFrame to CSV or other formats
- `save_text()`: Save text content (markdown, txt)
- `create_output_directory()`: Ensure output directory exists
- `generate_filename()`: Create descriptive filenames with timestamps

## Data Models

### PerformanceData

The core data structure representing model performance metrics.

```python
@dataclass
class PerformanceData:
    delay_type: str
    model: str
    auroc: float
    precision: float
    recall: float
    f1: float
    lead_time_mean: float
    lead_time_std: float
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate all fields are within acceptable ranges."""
        errors = []
        
        # Check delay type
        valid_delay_types = ['Language Delay', 'Motor Delay', 'ASD Indicators', 'ADHD Indicators']
        if self.delay_type not in valid_delay_types:
            errors.append(f"Invalid delay_type: {self.delay_type}")
        
        # Check model
        valid_models = ['LogReg', 'RF', 'GRU-FL', 'Trans-FL']
        if self.model not in valid_models:
            errors.append(f"Invalid model: {self.model}")
        
        # Check metric ranges
        for metric_name, metric_value in [
            ('auroc', self.auroc),
            ('precision', self.precision),
            ('recall', self.recall),
            ('f1', self.f1)
        ]:
            if not (0 <= metric_value <= 1):
                errors.append(f"{metric_name} must be between 0 and 1, got {metric_value}")
        
        # Check lead time
        if self.lead_time_mean < 0:
            errors.append(f"lead_time_mean must be positive, got {self.lead_time_mean}")
        if self.lead_time_std < 0:
            errors.append(f"lead_time_std must be positive, got {self.lead_time_std}")
        
        return len(errors) == 0, errors
```

### AnalysisResults

Container for computed analysis results.

```python
@dataclass
class AnalysisResults:
    summary_stats: pd.DataFrame
    best_models: Dict[str, str]
    fl_performance_gap: Dict[str, float]
    effect_sizes: pd.DataFrame
    confidence_intervals: pd.DataFrame
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'summary_stats': self.summary_stats.to_dict(),
            'best_models': self.best_models,
            'fl_performance_gap': self.fl_performance_gap,
            'effect_sizes': self.effect_sizes.to_dict(),
            'confidence_intervals': self.confidence_intervals.to_dict()
        }
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Required fields validation

*For any* input data structure, if validation passes, then all required fields (Delay Type, Model, AUROC, Precision, Recall, F1, Lead Time) must be present in the data.

**Validates: Requirements 1.1**

### Property 2: Metric range validation

*For any* validated performance data, all metric values must be within valid ranges: AUROC, Precision, Recall, and F1 must be in [0, 1], and Lead Time values must be positive.

**Validates: Requirements 1.2**

### Property 3: Input format equivalence

*For any* performance data, loading the same data from CSV, JSON, or dictionary format must produce equivalent internal DataFrames (same values, same structure).

**Validates: Requirements 1.3**

### Property 4: Validation error specificity

*For any* invalid data input, the validation error message must contain the specific field name and the nature of the validation failure.

**Validates: Requirements 1.4**

### Property 5: Color consistency across visualizations

*For any* set of visualizations generated from the same data, each model must be assigned the same color across all charts (e.g., LogReg always uses the same color).

**Validates: Requirements 2.2**

### Property 6: High-resolution image export

*For any* saved visualization, the output image file must have a DPI (dots per inch) of at least 300.

**Validates: Requirements 2.5**

### Property 7: Statistical computation accuracy

*For any* performance metric and delay type, the computed average across all models must equal the mean calculated directly from the raw data values.

**Validates: Requirements 3.1**

### Property 8: Best model identification correctness

*For any* delay type and metric, the identified best-performing model must have the maximum metric value among all models for that delay type.

**Validates: Requirements 3.3**

### Property 9: FL performance gap calculation

*For any* metric, the computed FL performance gap must equal the difference between the average of FL models (GRU-FL, Trans-FL) and the average of non-FL models (LogReg, RF).

**Validates: Requirements 4.3**

### Property 10: Lead time highlighting correctness

*For any* delay type, the model highlighted as having the longest lead time must have the maximum lead time mean value among all models for that delay type.

**Validates: Requirements 5.3**

### Property 11: Export format completeness

*For any* visualization export operation requesting multiple formats (PNG, PDF, SVG), all requested format files must be successfully created.

**Validates: Requirements 6.1**

### Property 12: Filename timestamp uniqueness

*For any* two export operations performed at different times (>1 second apart), the generated filenames must be unique due to timestamp inclusion.

**Validates: Requirements 6.3**

### Property 13: Dashboard panel completeness

*For any* dashboard generation, the output must include all required visualization types: model comparison charts, delay type analysis, and lead time visualizations.

**Validates: Requirements 7.1**

### Property 14: Effect size computation correctness

*For any* two models and a given metric, the computed effect size (Cohen's d) must equal (mean1 - mean2) / pooled_standard_deviation.

**Validates: Requirements 8.1**

### Property 15: Confidence interval bounds

*For any* metric and confidence level, the computed confidence interval must have a lower bound less than the mean and an upper bound greater than the mean.

**Validates: Requirements 8.2**

## Error Handling

### Validation Errors

- **Missing Required Fields**: Raise `ValueError` with specific field names
- **Invalid Metric Ranges**: Raise `ValueError` with field name and invalid value
- **Invalid Delay Type**: Raise `ValueError` with list of valid delay types
- **Invalid Model Name**: Raise `ValueError` with list of valid models

### File I/O Errors

- **File Not Found**: Raise `FileNotFoundError` with filepath
- **Permission Denied**: Raise `PermissionError` with filepath
- **Invalid Format**: Raise `ValueError` with format details

### Visualization Errors

- **Empty Data**: Raise `ValueError` indicating no data to plot
- **Invalid Metric**: Raise `ValueError` with list of valid metrics
- **Figure Save Failure**: Log warning and continue with other formats

### Recovery Strategies

1. **Partial Data**: If some rows fail validation, log warnings and continue with valid rows
2. **Export Failures**: If one format fails, continue with other formats
3. **Missing Optional Fields**: Use default values and log warnings

## Testing Strategy

### Unit Testing

The system will use `pytest` for unit testing with the following test categories:

1. **Data Loading Tests**
   - Test CSV parsing with valid data
   - Test JSON parsing with valid data
   - Test dictionary conversion
   - Test validation with invalid data (missing fields, out-of-range values)
   - Test error messages are descriptive

2. **Analysis Tests**
   - Test summary statistics computation
   - Test model comparison logic
   - Test FL performance gap calculation
   - Test best model identification
   - Test effect size calculations

3. **Visualization Tests**
   - Test figure creation (no exceptions)
   - Test color consistency across charts
   - Test label and title presence
   - Test output file creation

4. **Export Tests**
   - Test file creation in multiple formats
   - Test filename generation with timestamps
   - Test directory creation

### Property-Based Testing

The system will use `Hypothesis` for property-based testing. Each property-based test will run a minimum of 100 iterations.

1. **Property Test 1: Required fields validation**
   - **Feature: performance-analysis-visualization, Property 1: Required fields validation**
   - Generate random data structures with various combinations of fields
   - Verify validation passes only when all required fields are present

2. **Property Test 2: Metric range validation**
   - **Feature: performance-analysis-visualization, Property 2: Metric range validation**
   - Generate random metric values both within and outside valid ranges
   - Verify validation correctly accepts values in [0,1] for AUROC/Precision/Recall/F1 and positive values for Lead Time

3. **Property Test 3: Input format equivalence**
   - **Feature: performance-analysis-visualization, Property 3: Input format equivalence**
   - Generate random performance data
   - Load from CSV, JSON, and dict formats
   - Verify all produce equivalent DataFrames

4. **Property Test 4: Validation error specificity**
   - **Feature: performance-analysis-visualization, Property 4: Validation error specificity**
   - Generate invalid data with specific issues
   - Verify error messages contain the problematic field name

5. **Property Test 5: Color consistency across visualizations**
   - **Feature: performance-analysis-visualization, Property 5: Color consistency across visualizations**
   - Generate random performance data
   - Create multiple different visualizations
   - Verify each model uses the same color across all charts

6. **Property Test 6: High-resolution image export**
   - **Feature: performance-analysis-visualization, Property 6: High-resolution image export**
   - Generate random visualizations
   - Save to image files
   - Verify DPI >= 300 for all saved images

7. **Property Test 7: Statistical computation accuracy**
   - **Feature: performance-analysis-visualization, Property 7: Statistical computation accuracy**
   - Generate random performance data
   - Compute averages using the system
   - Verify results match direct pandas mean() calculations

8. **Property Test 8: Best model identification correctness**
   - **Feature: performance-analysis-visualization, Property 8: Best model identification correctness**
   - Generate random performance data
   - Identify best models for each delay type
   - Verify identified models have maximum metric values

9. **Property Test 9: FL performance gap calculation**
   - **Feature: performance-analysis-visualization, Property 9: FL performance gap calculation**
   - Generate random performance data
   - Compute FL performance gap
   - Verify gap equals (mean of FL models) - (mean of non-FL models)

10. **Property Test 10: Lead time highlighting correctness**
    - **Feature: performance-analysis-visualization, Property 10: Lead time highlighting correctness**
    - Generate random lead time data
    - Identify models with longest lead times
    - Verify identified models have maximum lead time values

11. **Property Test 11: Export format completeness**
    - **Feature: performance-analysis-visualization, Property 11: Export format completeness**
    - Generate random visualizations
    - Request multiple export formats (PNG, PDF, SVG)
    - Verify all requested files are created

12. **Property Test 12: Filename timestamp uniqueness**
    - **Feature: performance-analysis-visualization, Property 12: Filename timestamp uniqueness**
    - Generate multiple export operations with time delays
    - Verify all filenames are unique

13. **Property Test 13: Dashboard panel completeness**
    - **Feature: performance-analysis-visualization, Property 13: Dashboard panel completeness**
    - Generate random performance data
    - Create dashboard
    - Verify all required panels (model comparison, delay type analysis, lead time) are present

14. **Property Test 14: Effect size computation correctness**
    - **Feature: performance-analysis-visualization, Property 14: Effect size computation correctness**
    - Generate random performance data for two models
    - Compute effect size using the system
    - Verify result matches Cohen's d formula: (mean1 - mean2) / pooled_std

15. **Property Test 15: Confidence interval bounds**
    - **Feature: performance-analysis-visualization, Property 15: Confidence interval bounds**
    - Generate random performance data
    - Compute confidence intervals
    - Verify lower bound < mean < upper bound for all metrics

### Integration Testing

- Test end-to-end workflow: load data → analyze → visualize → export
- Test with the actual Table II data from the paper
- Test with edge cases (single model, single delay type)
- Test with large datasets (100+ rows)

### Test Configuration

- Minimum 100 iterations for each property-based test
- Use `pytest-cov` for coverage reporting (target: >90%)
- Use `pytest-benchmark` for performance testing
- Mock file I/O operations where appropriate to speed up tests
