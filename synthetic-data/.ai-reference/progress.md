# Progress Log

> Append progress logs conforming to this TypeScript schema:
interface stage_log {
    datetime: string; // stage start datetime YYYY-MM-DD HH-mm
    current_stage: string; // which stage is worked on?
    what_was_done: string[]; // what's implemented, artifacts produced, data generated?
    evidence: string[]; // prove what's done. ex osmo cmd output to show you have access to storage bucket
    decision: string[]; // what key decisions were made
    issues: string[]; // what issues were encountered, what's their status now
    next_stage: string;
}[]

```json
```
