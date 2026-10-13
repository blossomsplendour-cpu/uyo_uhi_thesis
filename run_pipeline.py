import os
import sys
from pathlib import Path

print('==================================================')
print('UYO UHI GEOSPATIAL PIPELINE - END-TO-END EXECUTION')
print('==================================================')

PROJECT_DIR = Path(__file__).parent.resolve()
data_dir = PROJECT_DIR / 'data' / 'outputs'
figures_dir = PROJECT_DIR / 'figures'

if not (data_dir / 'summary_by_date.csv').exists():
    print('Error: Master summary data not found. Please run Colab processing notebook.')
    sys.exit(1)

print('✓ Master data outputs verified.')
print(f'✓ Output directory: {data_dir}')
print(f'✓ Figures directory: {figures_dir}')
print('\nTo launch the interactive dashboard, run:')
print('  streamlit run dashboard/app.py')