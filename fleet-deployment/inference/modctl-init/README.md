# modctl Init Image

Bakes the [`modctl`](https://github.com/modelpack/modctl) CLI onto a small
Debian base so the `gr00t-inference` chart's `fetch-weights` init container can
pull the GR00T model weights from ACR.

## 🎯 Why it exists

The GR00T weights in ACR (`gr00t-n15-teradyne-dual-arm`) are a **CNCF ModelPack**
artifact: the manifest carries the `org.cncf.modctl.modelfile` annotation and the
weight layers use media type `application/vnd.cncf.model.weight.v1.raw` with an
`org.cncf.model.filepath` annotation. `oras pull` only materializes layers tagged
`org.opencontainers.image.title`, so it silently skips those weight layers and
nothing lands on disk. `modctl` understands the format and extracts the files
directly:

```bash
modctl pull <ref> --extract-dir <dir> --extract-from-remote
```

## 🔐 Auth

`modctl` loads registry credentials from the standard Docker `config.json`
(honoring `$DOCKER_CONFIG`), so the chart's existing projected ACR pull secret
works unchanged — no separate credential wiring.

## 🧱 Image

| Aspect      | Value                                                     |
|-------------|-----------------------------------------------------------|
| Base        | `debian:bookworm-slim`                                     |
| modctl      | `v0.2.1-cnai`, linux-arm64 binary (Jetson Orin / aarch64)  |
| Entry point | `modctl` (default `CMD ["--help"]`)                       |

## 🚀 Build

```bash
./build_and_push.sh 0.1
```

The registry is parameterized: `build_and_push.sh` reads `${REGISTRY}` (default
`immitationlearning.azurecr.io`). Optional positional args override the modctl
version and the base image:

```bash
REGISTRY=myregistry.azurecr.io ./build_and_push.sh 0.1 0.2.1-cnai debian:bookworm-slim
```

## ✅ Verify the pull (optional)

`verify-pod.yaml` is a standalone debug Pod that pulls the weights artifact to
`/tmp/out` and lists the extracted files, validating auth and extraction outside
the chart. Substitute your registry / pull secret / artifact ref, then:

```bash
kubectl apply -f verify-pod.yaml
kubectl logs -f pod/modctl-verify
kubectl delete -f verify-pod.yaml
```
