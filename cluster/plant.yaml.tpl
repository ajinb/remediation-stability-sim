# Queue-worker plant for the remediation-loop experiments.
# One copy per experiment slot; NS, NODEPORT and SERVICE_TIME placeholders are
# substituted by lab.py (string.Template, so shell dollars are doubled below).
#
#   jobs arrive (host producer, exact rate) -> Redis list "jobs" -> worker pods
#   each worker: BLPOP one job, hold it for SERVICE_TIME seconds, INCR stat:done
#
# The worker Deployment's replica count is the only actuator. Workers finish
# their current job on SIGTERM and exit, so a scale-down lands within about
# one service time instead of a 30 s grace period.
apiVersion: v1
kind: Namespace
metadata:
  name: ${NS}
  labels: {app.kubernetes.io/part-of: stability-lab}
---
apiVersion: apps/v1
kind: Deployment
metadata: {name: redis, namespace: ${NS}}
spec:
  replicas: 1
  selector: {matchLabels: {app: redis}}
  template:
    metadata: {labels: {app: redis}}
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        imagePullPolicy: IfNotPresent
        args: ["redis-server", "--save", "", "--appendonly", "no"]
        ports: [{containerPort: 6379}]
        resources:
          requests: {cpu: 50m, memory: 16Mi}
---
apiVersion: v1
kind: Service
metadata: {name: redis, namespace: ${NS}}
spec:
  selector: {app: redis}
  ports: [{port: 6379, targetPort: 6379}]
---
apiVersion: v1
kind: Service
metadata: {name: redis-host, namespace: ${NS}}
spec:
  type: NodePort
  selector: {app: redis}
  ports: [{port: 6379, targetPort: 6379, nodePort: ${NODEPORT}}]
---
apiVersion: apps/v1
kind: Deployment
metadata: {name: worker, namespace: ${NS}}
spec:
  replicas: 0
  selector: {matchLabels: {app: worker}}
  strategy:
    type: RollingUpdate
    rollingUpdate: {maxSurge: 100%, maxUnavailable: 100%}
  template:
    metadata: {labels: {app: worker}}
    spec:
      terminationGracePeriodSeconds: 5
      enableServiceLinks: false
      containers:
      - name: worker
        image: redis:7-alpine
        imagePullPolicy: IfNotPresent
        env: [{name: SERVICE_TIME, value: "${SERVICE_TIME}"}]
        command: ["/bin/sh", "-c"]
        args:
        - |
          stop=0
          trap 'stop=1' TERM
          while [ "$$stop" = 0 ]; do
            job=$$(redis-cli -h redis --raw BLPOP jobs 1 2>/dev/null)
            if [ -n "$$job" ]; then
              sleep "$$SERVICE_TIME"
              redis-cli -h redis INCR stat:done >/dev/null 2>&1
            fi
          done
        resources:
          requests: {cpu: 5m, memory: 6Mi}
          limits: {memory: 32Mi}
