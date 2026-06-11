# Synthetic Data Generation Workflow Blueprint

> Note: Work with the user to specify what the synthetic data generation
 workflow should look like, and record the outcome in the fields of this
 TypeScript interface:
interface workflow_blueprint {
    last_updated: string; // YYYY-MM-DD HH-mm
    workflow_type: string; // vda: video data augmentation | dig: defect image generation
    use_case: string; // ex. robot performing pick and place in a well-lit warehouse
    vda_requirements?: string[]; // required for vda workflow. ex. make background cluttered with dim lighting
    dig_requirements?: string[]; // required for dig workflow. ex. generate blue liquid spill in a red bin
    compute_cluster: string; // osmo cluster url
    input_data: string; // ex. the url or file path of the input video to augment
    notes: string; // anything to note about this feature
}

```json
```
