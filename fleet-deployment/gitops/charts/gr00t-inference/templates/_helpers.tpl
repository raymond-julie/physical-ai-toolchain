{{/*
Expand the name of the chart.
*/}}
{{- define "gr00t-inference.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Fully qualified app name.
*/}}
{{- define "gr00t-inference.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name (include "gr00t-inference.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{/*
Common labels.
*/}}
{{- define "gr00t-inference.labels" -}}
app.kubernetes.io/name: {{ include "gr00t-inference.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
{{- end -}}

{{/*
Selector labels.
*/}}
{{- define "gr00t-inference.selectorLabels" -}}
app.kubernetes.io/name: {{ include "gr00t-inference.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Full model reference: registry/repository[@digest|:tag].
Digest wins over tag when set.
*/}}
{{- define "gr00t-inference.modelRef" -}}
{{- if .Values.model.digest -}}
{{- printf "%s/%s@%s" .Values.registry .Values.model.repository .Values.model.digest -}}
{{- else -}}
{{- printf "%s/%s:%s" .Values.registry .Values.model.repository .Values.model.tag -}}
{{- end -}}
{{- end -}}

{{/*
Name of the PVC that caches model weights.
*/}}
{{- define "gr00t-inference.pvcName" -}}
{{- printf "%s-weights" (include "gr00t-inference.fullname" .) -}}
{{- end -}}

{{/*
Control UI fullname.
*/}}
{{- define "gr00t-inference.controlUi.fullname" -}}
{{- printf "%s-control-ui" (include "gr00t-inference.fullname" .) -}}
{{- end -}}

{{/*
Control UI selector labels.
*/}}
{{- define "gr00t-inference.controlUi.selectorLabels" -}}
app.kubernetes.io/name: {{ include "gr00t-inference.name" . }}-control-ui
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Robot client fullname.
*/}}
{{- define "gr00t-inference.robotClient.fullname" -}}
{{- printf "%s-robot-client" (include "gr00t-inference.fullname" .) -}}
{{- end -}}

{{/*
Robot client selector labels.
*/}}
{{- define "gr00t-inference.robotClient.selectorLabels" -}}
app.kubernetes.io/name: {{ include "gr00t-inference.name" . }}-robot-client
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
