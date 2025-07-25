# YouTube Data Collection for Weight Stigma Research

This repository contains the data collection pipeline for a research study investigating weight stigma in Brazilian YouTube content. The project uses the YouTube Data API v3 to collect and analyze video metadata and user comments.

## Overview

The research aims to understand patterns of weight stigma in social media by:
- Searching YouTube for videos using obesity-related keywords in Portuguese
- Collecting user comments from these videos
- Analyzing engagement patterns and content themes
- Providing anonymized datasets for further research

## Features

- **Systematic Data Collection**: Automated search and comment retrieval using YouTube API v3
- **Multiple API Key Support**: Handles quota limits with key rotation
- **Error Handling**: Robust error management and logging
- **Data Quality**: Built-in validation and cleaning processes
- **Privacy-Conscious**: Anonymization tools for ethical research
- **Publication-Ready**: Export functions for multiple data formats

## Setup

### Prerequisites

- Python 3.12 or higher
- YouTube Data API v3 key(s)
- Sufficient API quota for your research scope

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd paper_youtube_weight_stigma
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your YouTube API keys as environment variables:
```bash
export YOUTUBE_API_KEY_1='your_first_api_key'
export YOUTUBE_API_KEY_2='your_second_api_key'  # Optional
export YOUTUBE_API_KEY_3='your_third_api_key'   # Optional
```

### Getting YouTube API Keys

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the YouTube Data API v3
4. Create credentials (API key)
5. Restrict the key to YouTube Data API v3 for security

## Usage

### Quick Start

1. Open the Jupyter notebook:
```bash
jupyter notebook 01_get_data_api.ipynb
```

2. Run the configuration cell to verify your setup
3. Execute the data collection pipeline
4. Use the exploration functions to analyze your data

### Configuration

Modify the `Config` class in the notebook to customize:
- Search keywords
- Geographic and language targeting
- Date ranges
- File paths
- API limits


## Data Structure

The collected dataset includes:

- `video_id`: YouTube video identifier
- `textDisplay`: Comment text content
- `authorDisplayName`: Comment author (anonymized in publication exports)
- `publishedAt`: Comment publication timestamp
- `updatedAt`: Comment update timestamp
- `likeCount`: Number of likes on the comment
- `video_title`: Title of the YouTube video

## Ethical Considerations

This research follows ethical guidelines for social media research:

- **Public Data**: Only collects publicly available YouTube content
- **Anonymization**: Provides tools to anonymize personal identifiers
- **Terms of Service**: Complies with YouTube's terms of service
- **Rate Limiting**: Respects API quotas and rate limits
- **Data Minimization**: Collects only necessary data for research

### Important Notes

- Ensure compliance with your institution's IRB/ethics board
- Consider data protection regulations (GDPR, LGPD, etc.)
- Respect YouTube's terms of service and community guidelines
- Anonymize data before sharing or publication
- Document your methodology thoroughly

## File Structure

```
├── 01_get_data_api.ipynb     # Main data collection notebook
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── data/
│   ├── raw/                  # Raw collected data
│   ├── intermediate/         # Processed data
│   ├── outputs/              # Final datasets
│   └── tmp/                  # Temporary files
└── logs/                     # Application logs
```

## Output Formats

The pipeline supports multiple output formats:
- **Parquet**: Efficient binary format for large datasets
- **CSV**: Universal text format for broad compatibility
- **Excel**: For manual review and analysis
- **JSON**: Metadata and configuration files

## API Quota Management

YouTube API v3 has daily quotas. Tips for efficient usage:

- Use multiple API keys for larger studies
- The pipeline automatically rotates keys on quota exhaustion
- Monitor your usage in Google Cloud Console
- Consider collecting data over multiple days for large studies

## Troubleshooting

### Common Issues

1. **API Key Errors**: Verify keys are set correctly and have YouTube API access
2. **Quota Exceeded**: Wait 24 hours or use additional API keys
3. **Empty Results**: Check if videos have comments enabled
4. **Rate Limiting**: The pipeline handles this automatically

## Contributing

This is a research project. If you're using this code for your research:
1. Please cite the original paper (when published)
2. Follow ethical research practices
3. Consider sharing anonymized methodological improvements

## License

This project is licensed under MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
[Citation will be added upon publication]
```

## Contact

For questions about this research project, please contact elias.jacob@ufrn.br

## Acknowledgments

- YouTube Data API v3 for data access
- Brazilian research community for weight stigma awareness
- Open source Python ecosystem for research tools