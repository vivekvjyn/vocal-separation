# DDSP Vocal Separation

```mermaid
flowchart TD
    A[/mixed signal/] --> B[encoder]
    B --> C[harmonics]
    B --> D[noise]
    C --> E[reverb]
    D --> E
    E --> F[/target signal/]
```
