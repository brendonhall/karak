# karak

Automated mineralogy pipeline for SEM-EDS elemental map stacks. Takes jet-colormapped elemental map PNGs and produces mineral phase maps with chemical fingerprints.

## Installation

```bash
pip install karak
```

Or from source:

```bash
git clone https://github.com/brendonhall/karak.git
cd karak
pip install -e .
```

## Usage

```bash
karak                          # run full pipeline
karak --test-mode              # fast validation: 4 elements, 4x downsample
karak --from-stage denoise     # resume from a specific stage
karak --no-qc                  # skip QC figure generation
karak -c path/to/config.yaml   # use specific config file
karak -v                       # enable debug logging
```

## License

MIT
