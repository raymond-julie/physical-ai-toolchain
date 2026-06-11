# Synthetic Data Generation Infrastructure Blueprint

> Note: For now, we assume the OSMO cluster is up and running. No need
to read the rest of this file to provision a cluster.

> Note: Work with the admin user to specify the Azure infrastructure to
 run OSMO. Record the outcome in the fields of this TypeScript interface:
interface infra_blueprint {
    last_updated: string; // YYYY-MM-DD HH-mm
    azure_subscription: string;
    azure_resource_group: string;
    azure_region: string;
    cloud_storage: string; // azure://storage_name/container configured as the default profile bucket in osmo
    gpu_quota: string; // available GPU quota for the workload
    notes: string; // anything to note about the infra
}

```json
```
