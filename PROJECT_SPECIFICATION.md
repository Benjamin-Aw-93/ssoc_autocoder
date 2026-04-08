# SSOC Autocoder - Complete Reverse-Engineering Specification

> **Purpose:** This document captures every structural, visual, and functional detail needed for a future AI to recreate the SSOC Autocoder project pixel-perfect.

---

## 1. PROJECT OVERVIEW

### What It Does
The SSOC Autocoder predicts **Singapore Standard Occupational Classification (SSOC) 2020** codes from job descriptions posted on **MyCareersFuture (MCF)**. A user pastes a MCF job listing URL, the system fetches the job description, processes it through a custom hierarchical neural network built on DistilBERT, and returns the top 10 predicted SSOC codes with confidence scores.

### Team
- **Shaun Khoo** — co-developer
- **Benjamin Aw** — co-developer

### Timeline
| Phase | Period | Focus |
|-------|--------|-------|
| Phase 1 (completed) | Aug 2021 - Mar 2022 | Data processing, modelling, training, augmentation, POC deployment |
| Phase 2 (ongoing) | Apr 2022+ | Full deployment |

### External References
- Sprint tracking: ClickUp (project ID 3825139, space 43633588)
- Documentation/Architecture: Google Drive (folder `1a4zV5ILczikQnraScjXRwi2XCNEQtfwA`)
- AWS Console: account alias `project-finch`

---

## 2. SYSTEM ARCHITECTURE

```
                        +-----------------+
                        |   React SPA     |
                        | (port 3000)     |
                        +--------+--------+
                                 |
                         POST mcf_url
                                 |
                    +------------v-----------+
                    | Heroku CORS Proxy      |
                    | evening-plateau-95803  |
                    +------------+-----------+
                                 |
                    +------------v-----------+
                    | AWS API Gateway         |
                    | /predict?mcf_url=...    |
                    +------------+-----------+
                                 |
                   +-------------v--------------+
                   |  FastAPI / Lambda Handler   |
                   |  (Docker container)         |
                   +----+--------+---------+----+
                        |        |         |
              +---------v--+ +---v---+ +---v---------+
              | MCF API v2 | | Model | | Artifacts   |
              | (fetch JD) | | .pt   | | (encodings, |
              +------------+ +---+---+ | descriptions|
                                 |     +-------------+
                     +-----------v-----------+
                     | HierarchicalSSOC      |
                     | Classifier            |
                     | (DistilBERT + 5 heads)|
                     +-----------+-----------+
                                 |
                     Top 10 SSOC predictions
                     with confidence scores
```

### Data Flow
1. User enters MCF URL in the React frontend
2. Frontend calls the CORS proxy at Heroku which forwards to AWS
3. Backend extracts the UUID from the MCF URL using regex: `\-{1}([a-z0-9]{32})\?`
4. Backend calls MCF API: `GET https://api.mycareersfuture.gov.sg/v2/jobs/{uuid}`
5. Job title + description are extracted from the MCF response
6. Job description is processed through the NLP pipeline (HTML parsing, verb extraction, cleaning)
7. Processed text is tokenized with DistilBERT tokenizer and fed to the hierarchical classifier
8. Top 10 5-digit SSOC predictions with probabilities are returned
9. SSOC descriptions are looked up from `ssoc_desc.json`
10. Response is formatted and sent back to the frontend

---

## 3. FRONTEND SPECIFICATION

### 3.1 Tech Stack
| Dependency | Version | Purpose |
|-----------|---------|---------|
| react | 17.0.2 | UI framework |
| react-dom | 17.0.2 | DOM rendering |
| react-router-dom | 6.2.1 | Client-side routing |
| @mui/material | 5.3.0 | Primary component library |
| @mui/icons-material | 5.3.0 | Material icons |
| @mui/styles | 5.3.0 | Styling utilities |
| @material-ui/core | 4.12.3 | Legacy `makeStyles` API |
| @material-ui/icons | 4.11.2 | Legacy icons |
| @emotion/react | 11.7.1 | CSS-in-JS (MUI dependency) |
| @emotion/styled | 11.6.0 | Styled components (MUI dependency) |
| styled-components | 5.3.3 | Styled components |
| axios | 0.24.0 | HTTP client |
| react-html-parser | 2.0.2 | HTML string rendering |
| react-material-ui-carousel | 3.1.1 | Carousel (imported, usage unclear) |
| react-icons | 4.3.1 | Icon library |
| react-scroll | 1.8.4 | Smooth scrolling |
| react-scripts | 5.0.0 | CRA toolchain |
| Node.js | 16.x | Runtime |

### 3.2 Typography
- **Font Family:** `Roboto` from Google Fonts
- **Font Weights Loaded:**
  - `400 normal` (regular body text)
  - `700 italic` (bold italic)
- **Google Fonts URL:** `https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,400;1,700&display=swap`
- **Preconnect domains:**
  - `https://fonts.googleapis.com`
  - `https://fonts.gstatic.com` (crossorigin)

### 3.3 Color Palette

| Token | Hex | Usage |
|-------|-----|-------|
| Primary Button BG | `#ffb74d` | ColorButton background (Search, Search Again) |
| Primary Button Hover | `#fdd14d` | ColorButton hover state |
| Secondary Button BG | `#fff` | ColorButtonWhite background (Feeling Lucky) |
| Secondary Button Hover | `#fdd14d` | ColorButtonWhite hover |
| Card BG (light) | `#fafafa` | AccordianResult root, DescCard root |
| Result Card BG | `#eceff1` | ResultCard root background |
| Confidence: Low | `#ec1a1a` | Confidence < 33.33% |
| Confidence: Medium | `#ffb003` | Confidence 33.33% - 66.66% |
| Confidence: High | `#01bf71` | Confidence >= 66.66% |
| Theme Color | `#000000` | HTML meta theme-color |
| Background Color | `#ffffff` | Manifest background |

### 3.4 Component Tree

```
<React.StrictMode>
  <App>
    <BrowserRouter>
      <Home>                              (src/pages/index.js)
        <SearchSection />                  (always visible)
        <div ref={loadingScreenRef}>
          {isLoading ? <LoadingSection /> : null}
        </div>
        {!isEmpty(mainResult) ? <ResultsSection /> : null}
        {!isEmpty(errorMessage) ? <Snackbar> ... </Snackbar> : null}
      </Home>
    </BrowserRouter>
  </App>
</React.StrictMode>
```

### 3.5 Component: SearchSection

**File:** `src/components/SearchSection/index.js`

**Props:**
- `searchBarRef` — React ref for the TextField input
- `mcfUrl` — string state for the URL
- `setMcfUrl` — setter for mcfUrl
- `togglePress` — callback to initiate search

**Layout:**
```
<Container>                          // default MUI Container
  <Box height="60vh" sx={{ pt: "30%" }}>
    <Grid container rowSpacing={3} direction="row">

      {/* Title */}
      <Grid item xs={12}>
        <Typography variant="h3" gutterBottom align="center">
          <strong>SSOC Autocoder</strong>
        </Typography>
      </Grid>

      {/* Instructions */}
      <Grid item xs={12}>
        <Typography>
          Head over to the
          <Link target="_blank"
                href="https://www.mycareersfuture.gov.sg/"
                underline="always">
            MyCareersFuture
          </Link>
          website. Choose a job advertisment and paste the URL below:
        </Typography>
      </Grid>

      {/* Search Input */}
      <Grid item xs={12}>
        <TextField
          inputProps={{ inputMode: 'none' }}
          value={mcfUrl}
          inputRef={searchBarRef}
          fullWidth
          label="Paste URL of MyCareersFuture Job Ad here"
          onChange={e => setMcfUrl(e.target.value)}
        />
      </Grid>

      {/* Buttons */}
      <Grid item xs={12}>
        <Grid container spacing={2} align="center">
          <Grid item xs={1} />           {/* left spacer */}
          <Grid item xs={5}>
            <ColorButton onClick={togglePress}>Search</ColorButton>
          </Grid>
          <Grid item xs={5}>
            <ColorButtonWhite variant="outlined">
              Feeling lucky
            </ColorButtonWhite>
          </Grid>
          <Grid item xs={1} />           {/* right spacer */}
        </Grid>
      </Grid>

    </Grid>
  </Box>
</Container>
```

**Styled Components:**
```javascript
// ColorButton (primary)
const ColorButton = styled(Button)(({ theme }) => ({
  color: theme.palette.getContrastText('#ffb74d'),
  backgroundColor: '#ffb74d',
  '&:hover': { backgroundColor: '#fdd14d' }
}));

// ColorButtonWhite (secondary)
const ColorButtonWhite = styled(Button)(({ theme }) => ({
  color: theme.palette.getContrastText('#ffb74d'),
  backgroundColor: '#fff',
  '&:hover': { backgroundColor: '#fdd14d' }
}));
```

### 3.6 Component: LoadingSection

**File:** `src/components/LoadingSection/index.js`

**Props:** none

**Layout:**
```
<Container>
  <Box height="100vh" sx={{ pt: "35%" }}>
    <Grid container rowSpacing={3} direction="row">
      <Grid item xs={12} align="center">
        <CircularProgress size="10rem" />
      </Grid>
      <Grid item xs={12} align="center">
        <Typography variant="h5">
          Give us a moment to generate the predictions...
        </Typography>
      </Grid>
    </Grid>
  </Box>
</Container>
```

### 3.7 Component: ResultsSection

**File:** `src/components/ResultsSection/index.js`

**Props:**
- `executeSearchScroll` — callback to scroll back to search and clear state
- `mainResult` — API response object

**Layout:**
```
<Container maxWidth="xl">
  <Box height="100vh" sx={{ py: '5rem' }}>
    <Grid container spacing={3} alignItems="flex-start" justifyContent="center">
      <Grid item xs={12}>
        <Stack direction="column"
               divider={<Divider orientation="horizontal" flexItem sx={{ my: 2 }} />}>

          <DescCard mainResult={mainResult} />
          <ResultCard mainResult={mainResult} />

          <Grid container spacing={2} alignItems="center" justifyContent="center">
            <Grid item xs={12}>
              <Typography align="left">
                Next 9 predictions (in order) are:
              </Typography>
            </Grid>

            {mainResult.other_predictions?.map((nextpred, i) => (
              <Grid item xs={12} key={i}>
                <AccordianResult results={nextpred} idx={i} />
              </Grid>
            ))}

            <Grid item>
              <ColorButton
                endIcon={<SettingsBackupRestoreIcon />}
                onClick={executeSearchScroll}>
                Click here to search for another entry
              </ColorButton>
            </Grid>
          </Grid>

        </Stack>
      </Grid>
    </Grid>
  </Box>
</Container>
```

**Styled Component (ColorButton):**
```javascript
const ColorButton = styled(Button)(({ theme }) => ({
  color: theme.palette.getContrastText('#ffb74d'),
  backgroundColor: '#ffb74d',
  '&:hover': { backgroundColor: '#fdd14d' }
}));
```

### 3.8 Component: ResultCard

**File:** `src/components/ResultCard/index.js`

**Props:**
- `mainResult` — API response object (uses `mainResult.top_prediction`)

**Styles (makeStyles):**
```javascript
root: {
  borderRadius: 12,
  minWidth: 256,
  textAlign: 'center',
  backgroundColor: '#eceff1'
},
header: { textAlign: 'center', spacing: 10 },
list: { padding: '20px' },
action: { display: 'flex', justifyContent: 'space-around' }
```

**Color-Coding Logic:**
```javascript
const colorCode = (value) => {
  const parsedValue = parseFloat(value);
  if (parsedValue < 33.33) return '#ec1a1a';    // red
  else if (parsedValue < 66.66) return '#ffb003'; // amber
  return '#01bf71';                                // green
};
```

**Layout:**
```
<Card elevation={5} className={classes.root}>
  <CardHeader title={<strong>Top Prediction</strong>} className={classes.header} />
  <Divider variant="middle" />
  <CardContent>
    <Typography noWrap variant="h5" gutterBottom align="left">
      <strong>SSOC Title:</strong> {mainResult.top_prediction.SSOC_Title}
    </Typography>
    <Typography noWrap variant="h5" gutterBottom align="left">
      <strong>SSOC Code:</strong> {mainResult.top_prediction.SSOC_Code}
    </Typography>
    <Typography noWrap variant="h5" gutterBottom align="left"
                color={colorCode(mainResult.top_prediction.Prediction_Confidence)}>
      <strong>Confidence:</strong> {mainResult.top_prediction.Prediction_Confidence}
    </Typography>
    <Divider variant="middle" sx={{ my: 2 }} />
    <Typography noWrap variant="h5" gutterBottom align="center">
      <strong>SSOC Description:</strong>
    </Typography>
    <Box component="div" sx={{ maxHeight: '850px', overflow: 'auto', m: '1rem' }}>
      <Typography align="left">
        {mainResult.top_prediction.SSOC_Description}
      </Typography>
    </Box>
  </CardContent>
</Card>
```

### 3.9 Component: DescCard

**File:** `src/components/DescCard/index.js`

**Props:**
- `mainResult` — API response object (uses `mainResult.mcf_job_title`, `mainResult.mcf_job_desc`)

**Styles (makeStyles):**
```javascript
root: {
  borderRadius: 12,
  minWidth: 256,
  textAlign: 'center',
  backgroundColor: '#fafafa'
},
header: { textAlign: 'center', spacing: 10 },
list: { padding: '20px' },
action: { display: 'flex', justifyContent: 'space-around' },
dropdown: {
  fontSize: theme.typography.pxToRem(15),
  color: theme.palette.text.secondary,
  marginLeft: 'auto'
}
```

**Expand/Collapse Animation:**
```javascript
const ExpandMore = styled((props) => {
  const { expand, ...other } = props;
  return <IconButton {...other} />;
})(({ theme, expand }) => ({
  transform: !expand ? 'rotate(0deg)' : 'rotate(180deg)',
  marginLeft: 'auto',
  transition: theme.transitions.create('transform', {
    duration: theme.transitions.duration.shortest
  })
}));
```

**Layout:**
```
<Card className={classes.root}>
  <CardHeader title={<strong>Job Description</strong>} className={classes.header} />
  <Divider variant="middle" />
  <CardContent>
    <Typography noWrap variant="h8" align="left">
      <strong>Job Title:</strong> {mainResult.mcf_job_title || 'xxx'}
    </Typography>
  </CardContent>
  <CardActions disableSpacing>
    {expanded
      ? <Typography className={classes.dropdown}>Hide Job Description</Typography>
      : <Typography className={classes.dropdown}>Expand to show Job Description</Typography>
    }
    <ExpandMore expand={expanded} onClick={handleExpandClick}>
      <ExpandMoreIcon />
    </ExpandMore>
  </CardActions>
  <Collapse in={expanded} timeout="auto" unmountOnExit>
    <CardContent>
      <Typography paragraph>
        <Typography align="left">
          {ReactHtmlParser(mainResult.mcf_job_desc) || 'xxx'}
        </Typography>
      </Typography>
    </CardContent>
  </Collapse>
</Card>
```

### 3.10 Component: AccordianResult

**File:** `src/components/AccordianResult/index.js`

**Props:**
- `results` — single prediction object `{ SSOC_Title, SSOC_Code, Prediction_Confidence, SSOC_Description }`
- `idx` — zero-based index (displayed as `idx + 2` for ranking)

**Styles (makeStyles):**
```javascript
root: { backgroundColor: '#fafafa' },
accordianHeading: {
  fontSize: theme.typography.pxToRem(15),
  flexBasis: '10%',
  flexShrink: 0,
  textAlign: 'left',
  flexGrow: 0
},
accordianSecondaryHeading: {
  fontSize: theme.typography.pxToRem(15),
  textAlign: 'left'
},
accordianThirdHeading: {
  fontSize: theme.typography.pxToRem(15),
  color: theme.palette.text.secondary,
  textAlign: 'center',
  flexBasis: '20%',
  flexShrink: 0,
  flexGrow: 0
}
```

**Layout:**
```
<Accordion className={classes.root}
           expanded={expanded === 'panel1'}
           onChange={handleChange('panel1')}>
  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
    <Typography className={classes.accordianHeading}>{idx + 2}</Typography>
    <div style={{ overflow: "hidden", textOverflow: "ellipsis", width: '15rem' }}>
      <Typography noWrap className={classes.accordianSecondaryHeading}>
        {results.SSOC_Title}
      </Typography>
    </div>
    <Typography className={classes.accordianThirdHeading}>
      [{results.Prediction_Confidence}]
    </Typography>
  </AccordionSummary>
  <AccordionDetails>
    <Typography><strong>SSOC Title:</strong> {results.SSOC_Title}</Typography>
    <Typography><strong>SSOC Code:</strong> {results.SSOC_Code}</Typography>
    <Typography><strong>Confidence:</strong> {results.Prediction_Confidence}</Typography>
    <Divider variant="middle" sx={{ my: 2 }} />
    <Typography><strong>SSOC Description:</strong></Typography>
    <Typography>{results.SSOC_Description}</Typography>
  </AccordionDetails>
</Accordion>
```

### 3.11 Component: Home (Page Controller)

**File:** `src/pages/index.js`

**State Management:**
```javascript
const [mcfUrl, setMcfUrl] = useState("");
const [mainResult, setmainResult] = useState({});
const [isLoading, setisLoading] = useState(false);
const [isError, setisError] = useState(false);
const [errorMessage, seterrorMessage] = useState({});
```

**Refs:**
- `loadingScreenRef` — scrolls to loading section on state change
- `searchBarRef` — focuses back on search input when "search again" is clicked

**Key Behavior:**
- `useEffect` on `isLoading` triggers smooth scroll to loading screen
- `togglePress`: clears results -> sets loading -> calls API -> sets results -> clears loading
- `executeSearchScroll`: focuses search bar, clears URL + results (acts as reset)
- Error handling via MUI `Snackbar` + `Alert` with severity "error", autoHideDuration 6000ms
- Error display: `Error {errorMessage.status}!` title + `{errorMessage.data.Error}` body

### 3.12 API Integration

**File:** `src/components/API/lambdaAPI.js`

```javascript
const URL = 'https://evening-plateau-95803.herokuapp.com/' +
            'https://e81tvuwky6.execute-api.us-east-1.amazonaws.com/predict';

const getSSOCData = async (mcfurl) => {
  const data = await axios.get(URL, {
    params: { 'mcf_url': mcfurl }
  });
  return data;
};
```

### 3.13 API Response Shape

```json
{
  "mcf_job_id": "JOB-2022-XXXXXXX",
  "mcf_job_title": "Data Scientist",
  "mcf_job_desc": "<p>HTML job description...</p>",
  "top_prediction": {
    "SSOC_Code": "21121",
    "SSOC_Title": "Data Scientist",
    "SSOC_Description": "Data scientists apply...",
    "Prediction_Confidence": "87.23%"
  },
  "other_predictions": [
    {
      "SSOC_Code": "21129",
      "SSOC_Title": "...",
      "SSOC_Description": "...",
      "Prediction_Confidence": "5.12%"
    }
    // ... 8 more items (9 total)
  ]
}
```

### 3.14 HTML Template

**File:** `Deployments/frontend/public/index.html`
```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <link rel="icon" href="%PUBLIC_URL%/favicon.ico" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="Web site created using create-react-app" />
    <link rel="apple-touch-icon" href="%PUBLIC_URL%/logo192.png" />
    <link rel="manifest" href="%PUBLIC_URL%/manifest.json" />
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:ital,wght@0,400;1,700&display=swap"
          rel="stylesheet">
    <title>SSOC AutoCoder</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
```

### 3.15 Manifest

```json
{
  "short_name": "SSOC AutoCoder",
  "name": "Predicting SSOC values from MCF job description",
  "icons": [
    { "src": "favicon.ico", "sizes": "64x64 32x32 24x24 16x16", "type": "image/x-icon" },
    { "src": "logo192.png", "type": "image/png", "sizes": "192x192" },
    { "src": "logo512.png", "type": "image/png", "sizes": "512x512" }
  ],
  "start_url": ".",
  "display": "standalone",
  "theme_color": "#000000",
  "background_color": "#ffffff"
}
```

### 3.16 Docker (Frontend)

```dockerfile
FROM node:16.15.1-buster
WORKDIR /code
COPY package.json package-lock.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["npm", "run", "start"]
```

---

## 4. BACKEND SPECIFICATION

### 4.1 FastAPI Backend (Amplify Deployment)

**File:** `Deployments/backend/amplify/backend/api/modelpredict/src/app/main.py`

**Framework:** FastAPI

**Endpoints:**
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check — returns `"SSOC Autocoder API is active and ready."` |
| GET | `/predict?mcf_url={url}` | Main prediction endpoint |

**Startup Initialization (module-level):**
```python
model, tokenizer, encoding = initialise()
ssoc_desc = json.load(open('artifacts/ssoc_desc.json'))
```

**`initialise()` function:**
- Loads model from `artifacts/ssoc-autocoder-model.pickle` (pickle format)
- Loads tokenizer from `artifacts/distilbert-tokenizer-pretrained-7epoch`
- Loads SSOC encoding from `artifacts/ssoc-idx-encoding.json`

**`/predict` flow:**
1. `return_uuid(mcf_url)` — extracts UUID via regex or returns random dummy UUID if `mcf_url == 'feelinglucky'`
2. `call_mcf_api(mcf_uuid)` — `GET https://api.mycareersfuture.gov.sg/v2/jobs/{uuid}`
3. `generate_predictions(model, tokenizer, encoding, mcf_url, resp, ssoc_desc)`:
   - Extracts `jobPostId`, `title`, `description` from MCF response
   - If "feelinglucky": returns pre-computed dummy data with SSOC descriptions appended
   - Else: processes description through `processing.process_text()`, generates predictions via `model_prediction.generate_single_prediction()`
   - Formats top 10 5D predictions with SSOC code, title, description, confidence percentage

**UUID Extraction Regex:**
```python
re.search('\\-{1}([a-z0-9]{32})\\?', mcf_url + "?")
```

**Error Handling:**
- Invalid URL: HTTP 404 with detail message
- MCF API failure: HTTP 404 with detail message

### 4.2 Lambda Backend

**File:** `Deployments/lambda/model-predict/app.py`

Same logic as FastAPI but adapted for AWS Lambda:
- `handler(event, context)` — entry point
- Reads `mcf_url` from `event['queryStringParameters']['mcf_url']`
- Returns `{ 'statusCode': 200, 'body': json.dumps(output) }`
- Error responses use `statusCode: 400`

**Key difference:** Lambda variant calls `model_prediction.model_predict()` with file paths directly rather than pre-loading.

### 4.3 Lambda Dummy API

**File:** `Deployments/lambda/dummy-api/lambda_function.py`

A simpler Lambda that serves pre-computed dummy data (for testing/demo purposes).

### 4.4 Backend Artifacts

| File | Content |
|------|---------|
| `ssoc-autocoder-model.pickle` | Serialized PyTorch model (HierarchicalSSOCClassifier) |
| `distilbert-tokenizer-pretrained-7epoch/` | Fine-tuned DistilBERT tokenizer (vocab.txt, tokenizer_config.json, special_tokens_map.json) |
| `ssoc-idx-encoding.json` | SSOC-to-index and index-to-SSOC mappings for all 5 levels |
| `ssoc_desc.json` | SSOC code -> { title, description } lookup |
| `dummy_data.json` | 50 pre-computed prediction results for "feeling lucky" mode |

### 4.5 Backend Docker

```dockerfile
FROM public.ecr.aws/lambda/python:3.8
# (Lambda container image for model-predict)
```

### 4.6 Backend Requirements

```
torch==1.10.0
transformers==4.15.0
fastapi==0.70.0
uvicorn==0.15.0
urllib3
```

---

## 5. ML MODEL SPECIFICATION

### 5.1 Architecture: HierarchicalSSOCClassifier

The model uses a **cascading hierarchical classification** approach. DistilBERT generates 768-dim embeddings, then 5 separate classification stacks predict SSOC at each digit level. Each subsequent stack receives the previous level's predictions concatenated to the embeddings.

```
Input Text
    |
    v
DistilBERT (frozen, pretrained on MCF JDs)
    |
    v
768-dim [CLS] embedding (X)
    |
    +---> Stack 1 (768 -> 9 classes)          --> SSOC_1D prediction
    |     X_2 = concat(X, SSOC_1D_pred)
    |
    +---> Stack 2 (777 -> 42 classes)         --> SSOC_2D prediction
    |     X_3 = concat(X_2, SSOC_2D_pred)
    |
    +---> Stack 3 (819 -> 144 classes)        --> SSOC_3D prediction
    |     X_4 = concat(X_3, SSOC_3D_pred)
    |
    +---> Stack 4 (963 -> 413 classes)        --> SSOC_4D prediction
    |     X_5 = concat(X_4, SSOC_4D_pred)
    |
    +---> Stack 5 (1376 -> 997 classes)       --> SSOC_5D prediction
```

### 5.2 SSOC Classification Levels

| Level | Digits | Number of Classes | Input Dimensions |
|-------|--------|-------------------|------------------|
| SSOC_1D | 1 | 9 | 768 |
| SSOC_2D | 2 | 42 | 768 + 9 = 777 |
| SSOC_3D | 3 | 144 | 777 + 42 = 819 |
| SSOC_4D | 4 | 413 | 819 + 144 = 963 |
| SSOC_5D | 5 | 997 | 963 + 413 = 1376 |

### 5.3 Stack Architecture (V1)

Each stack follows the pattern `Linear -> ReLU -> Dropout(0.3)` repeated, with progressive dimensionality reduction:

**Stack 1 (1D, 768 -> 9):**
```
Linear(768, 768) -> ReLU -> Dropout(0.3)
Linear(768, 768) -> ReLU -> Dropout(0.3)
Linear(768, 128) -> ReLU -> Dropout(0.3)
Linear(128, 9)
```

**Stack 2 (2D, 777 -> 42):**
```
Linear(777, 777) -> ReLU -> Dropout(0.3)
Linear(777, 777) -> ReLU -> Dropout(0.3)
Linear(777, 256) -> ReLU -> Dropout(0.3)
Linear(256, 128) -> ReLU -> Dropout(0.3)
Linear(128, 42)
```

**Stack 3 (3D, 819 -> 144):**
```
Linear(819, 819) -> ReLU -> Dropout(0.3)
Linear(819, 819) -> ReLU -> Dropout(0.3)
Linear(819, 819) -> ReLU -> Dropout(0.3)
Linear(819, 512) -> ReLU -> Dropout(0.3)
Linear(512, 256) -> ReLU -> Dropout(0.3)
Linear(256, 144)
```

**Stack 4 (4D, 963 -> 413):**
```
Linear(963, 963) -> ReLU -> Dropout(0.3)
Linear(963, 963) -> ReLU -> Dropout(0.3)
Linear(963, 963) -> ReLU -> Dropout(0.3)
Linear(963, 768) -> ReLU -> Dropout(0.3)
Linear(768, 512) -> ReLU -> Dropout(0.3)
Linear(512, 413)
```

**Stack 5 (5D, 1376 -> 997):**
```
Linear(1376, 1376) -> ReLU -> Dropout(0.3)
Linear(1376, 1376) -> ReLU -> Dropout(0.3)
Linear(1376, 1376) -> ReLU -> Dropout(0.3)
Linear(1376, 1376) -> ReLU -> Dropout(0.3)
Linear(1376, 997)
```

### 5.4 Prediction Parameters

```python
ssoc_prediction_parameters = {
    'SSOC_1D': {'top_n': 2,  'min_prob': 0.5},
    'SSOC_2D': {'top_n': 5,  'min_prob': 0.4},
    'SSOC_3D': {'top_n': 5,  'min_prob': 0.3},
    'SSOC_4D': {'top_n': 5,  'min_prob': 0.2},
    'SSOC_5D': {'top_n': 10, 'min_prob': 0.1}
}
```

### 5.5 Prediction Generation

1. Tokenize text with DistilBERT tokenizer (`max_length=512`, `padding='max_length'`, `truncation=True`)
2. Extract `input_ids` and `attention_mask` as tensors
3. Set model to `eval()` mode, run forward pass with `torch.no_grad()`
4. Apply `Softmax(dim=1)` to get probabilities
5. For each SSOC level: sort predictions descending, take top_n, convert indices to SSOC codes via encoding
6. Return `{ predicted_ssoc: [...], predicted_proba: [...], accurate_prediction: bool|None }`

### 5.6 Pre-training: Masked Language Model

**File:** `ssoc_autocoder/run_mlm.py` (HuggingFace script, PyTorch)
**File:** `ssoc_autocoder/masked_language_model.py` (Custom TensorFlow implementation)

Two implementations exist:

**PyTorch (run_mlm.py):**
- Standard HuggingFace `Trainer`-based MLM fine-tuning
- Supports line-by-line or concatenated text processing
- MLM probability: 15% (default)
- Batch size: 2 (hardcoded override)
- Uses `DataCollatorForLanguageModeling`

**TensorFlow (masked_language_model.py):**
- Custom implementation with whole-word masking support
- Two masking modes: `'normal masking'` and `'whole word masking'`
- Configurable chunk size, train/eval split, learning rate, warmup, weight decay
- CLI arguments for all parameters
- Evaluates perplexity before and after training
- Saves model with `save_pretrained()`

**Pre-trained tokenizer:** `distilbert-tokenizer-pretrained-7epoch` (7 epochs of MLM on MCF data)

### 5.7 Model Variants

The codebase contains two classifier versions:
- `HierarchicalSSOCClassifier_V1` — DistilBERT base frozen, stacks trainable
- `HierarchicalSSOCClassifier_V2` — exists in model_training.py (extended version)

Both use an alias `HierarchicalSSOCClassifier` for deployment.

---

## 6. DATA PROCESSING PIPELINE

### 6.1 Main Entry Point

**File:** `ssoc_autocoder/processing.py`

```python
process_text(raw_text) -> str
```

**Flow:**
1. `clean_raw_string(text)` — removes `\n`, `\xa0`, `\t`, special chars; replaces `No.` with `Number`
2. If total word count < 100: return `final_cleaning(text)` (too short to process)
3. Try extraction in priority order:
   - `process_li_tag(text)` — extracts `<ol>/<ul>` lists
   - `process_p_list(text)` — extracts `<p>` tags with bullet points (`\u2022`, `\u002d`, `\u00b7`) or numbers
   - `process_p_tag(text)` — extracts `<p>` tags starting with verbs
4. If extracted text is < 50 words: fall back to cleaned original text
5. Apply `final_cleaning()` to the result

### 6.2 Verb Detection

```python
check_if_first_word_is_verb(string, nlp) -> bool
```

- Uses spaCy `en_core_web_lg` for POS tagging
- **Override FALSE list:** `['proven', 'possess']`
- **Override TRUE list:** `['review', 'responsible', 'design', 'to', 'able']`
- Strips `'you are'` / `'are you'` prefix before checking
- Falls back to `nlp(string)[0].pos_ == 'VERB'`

### 6.3 List Verb Scoring (`check_list_for_verbs`)

For each candidate list:
1. Count items where first word is a verb
2. Compute verb_score = count / total items
3. Merge short lists (< 6 items) with verb_score >= 70% into the preceding list
4. Recurse if merges occurred
5. Return the list with highest verb score (must be >= 50%)
6. For very short lists (< 3 items), cap verb score at 50%

### 6.4 Final Text Cleaning (`final_cleaning`)

Sequential regex operations:
1. `<br>` -> `.`
2. `&amp;` -> `&`
3. `&nbsp;` -> ` `
4. `&rsquo;`, `&lsquo;` -> `'`
5. `&ldquo;`, `&rdquo;` -> `'`
6. Unicode curly quotes -> `'`
7. Bullet points/numbers after `>` -> `.`
8. Non-ASCII chars -> `.`
9. Closing HTML tags -> `.`
10. Opening HTML tags -> ` `
11. `;` -> `.`
12. Normalize whitespace
13. Remove leading punctuation
14. Collapse consecutive periods
15. Remove space before punctuation
16. Remove punctuation after punctuation
17. Replace `.` before lowercase with ` ` (likely mid-sentence)
18. Filter out single-word segments
19. Trim and add final period

### 6.5 MCF JSON Extraction

**File:** `ssoc_autocoder/converting_json.py`

`extract_mcf_data(json)` extracts:
- General: `uuid`, `title`, `description`, `minimumYearsExperience`, `numberOfVacancies`
- Skills: joined comma-separated string
- Company: `name`, `description`, `ssicCode`, `employeeCount` (from `hiringCompany` or `postedCompany`)
- Metadata: `originalPostingDate`, `newPostingDate`, `expiryDate`, `totalNumberOfView`, `totalNumberJobApplication`
- Salary: `maximum`, `minimum`, `type.id`, `type.salaryType`

`extract_and_split(path)` processes all JSON files in a directory and groups by `{year}-{week_number}`.

### 6.6 Data Augmentation

**File:** `ssoc_autocoder/augmentation.py`

Six augmentation techniques (via `nlpaug`):

| Technique | Model/Source | Action |
|-----------|-------------|--------|
| Word Embedding | GloVe 840B 300d | substitute (aug_p=0.5, aug_min=10) |
| Back Translation | facebook/wmt19-en-de, facebook/wmt19-de-en | translate round-trip |
| Synonym | PPDB 2.0 TLDR | substitute (aug_p=0.5, aug_min=10) |
| Contextual Embedding | distilbert-base-uncased | substitute (top_k=5) |
| Sentence Augmentation | distilgpt2 | append (min_length=50) |
| Summarization | t5-base | abstractive summary |

**Random Phrase Injection:**
24 common job posting phrases randomly inserted into augmented text:
- "We are looking for/searching for a candidate who is"
- "In this role, you will be responsible for"
- "Top skills and proficiencies include"
- etc.

---

## 7. PYTHON PACKAGE STRUCTURE

### Package: `ssoc_autocoder/`

```
ssoc_autocoder/
  __init__.py              # exports: ['processing', 'train', 'predict']
  processing.py            # Job description HTML parsing and text extraction
  features.py              # (empty file — placeholder)
  utils.py                 # verboseprint(), processing_raw_data()
  converting_json.py       # MCF JSON to structured data extraction
  augmentation.py          # NLP-based data augmentation (6 techniques)
  model_training.py        # Dataset class, model architectures (V1, V2), training loop
  model_prediction.py      # Prediction generation (single and multiple)
  predict.py               # Standalone prediction script (interactive)
  generate_prediction.py   # (empty file — placeholder)
  masked_language_model.py # TensorFlow MLM pre-training pipeline
  run_mlm.py               # PyTorch MLM pre-training (HuggingFace Trainer)
  run_mlm_aws.py           # AWS-adapted MLM training variant
  model_training_aws.py    # AWS-adapted model training variant
  requirements.txt         # Package-level dependencies
```

### Module Function Inventory

**`processing.py`:**
- `remove_prefix(text, prefixes)` -> str
- `check_if_first_word_is_verb(string, nlp)` -> bool
- `clean_html_unicode(string)` -> str
- `check_list_for_verbs(list_elements, nlp)` -> list/BS4Tag
- `process_li_tag(text, nlp)` -> list
- `process_p_list(text, nlp)` -> list
- `process_p_tag(text, nlp)` -> str
- `clean_raw_string(string)` -> str
- `final_cleaning(processed_text)` -> str
- `text_length_less_than(text, length)` -> bool
- `process_text(raw_text)` -> str

**`utils.py`:**
- `verboseprint(verbose=True)` -> function
- `processing_raw_data(filename, *colnames)` -> DataFrame

**`converting_json.py`:**
- `extract_mcf_data(json)` -> (dict, str)
- `extract_and_split(path)` -> dict

**`augmentation.py`:**
- `trim(text, max_len)` -> str
- `data_aug_collated(text, params)` -> dict
- `random_insert(text, common_phrases, prob, edit_phrase)` -> str
- `data_augmentation(text, prob, edit_phrase, params, common_phrases)` -> dict

**`model_training.py`:**
- `generate_encoding(reference_data, ssoc_colname)` -> dict
- `import_ssoc_idx_encoding(filename)` -> dict
- `encode_dataset(data, encoding, colnames)` -> DataFrame
- `prepare_data(encoded_train, encoded_test, tokenizer, colnames, parameters)` -> (DataLoader, DataLoader)
- Class `SSOC_Dataset(Dataset)` — PyTorch Dataset
- Class `HierarchicalSSOCClassifier_V1(nn.Module)` — main model
- Class `HierarchicalSSOCClassifier_V2(nn.Module)` — variant

**`model_prediction.py`:**
- `model_predict(pretrained_filepath, model_filepath, tokenizer_filepath, ssoc_idx_encoding_filepath, text)` -> dict
- `generate_single_prediction(model, tokenizer, text, target, encoding, ssoc_prediction_parameters)` -> dict
- `generate_multiple_predictions(model, tokenizer, test_set, encoding, ssoc_prediction_parameters, ssoc_level)` -> list

**`masked_language_model.py`:**
- `tokenize_function(dataset, tokenizer)` -> dict
- `group_texts(tokenized_dataset, chunk_size)` -> dict
- `whole_word_masking_data_collator(features, whole_word_masking_probability)` -> batch
- `split_dataset(train_size, fraction, grouped_tokenized_datasets, seed)` -> dataset
- `masking(downsampled_dataset, function, batch_size, split, type_of_masking)` -> tf.Dataset
- `get_metrics(model, tf_eval_dataset)` -> float (perplexity)
- `predict_masked_word(text, new_model)` -> list
- `trainer(tf_train_dataset, model, lr, warmup, wdr)` -> model
- `main(...)` — full training pipeline

---

## 8. DEPLOYMENT

### 8.1 AWS Amplify (Primary)

**Backend stack:**
- AWS Amplify CLI project
- REST API: `modelpredict`
- Docker container deployed via Amplify
- CloudFormation template at `amplify/backend/api/modelpredict/modelpredict-cloudformation-template.json`
- Build spec: `buildspec.yml`

**Container:**
```dockerfile
FROM public.ecr.aws/lambda/python:3.8
# Model artifacts bundled in image
```

### 8.2 AWS Lambda (Alternative)

- `model-predict/` — full prediction Lambda with Docker
- `dummy-api/` — lightweight dummy data Lambda (packaged as .zip)

### 8.3 Data Processing Feature Deployment

**Files in `Deployments/feature/add-data-processing/`:**
- `get_json.py` — fetch MCF data
- `json_object_from_mcf_job_id.py` — extract JSON from MCF job IDs
- `json_to_processed_csv.py` — JSON -> processed CSV
- `json_to_raw_csv.py` — JSON -> raw CSV
- `raw_csv_to_processed_csv.py` — raw CSV -> processed CSV

### 8.4 Infrastructure

- **AWS Account:** `project-finch`
- **Region:** `us-east-1`
- **CORS Proxy:** Heroku app `evening-plateau-95803`
- **API Gateway:** `e81tvuwky6.execute-api.us-east-1.amazonaws.com`

---

## 9. TESTING

### Test Files

```
Tests/
  test_processing.py        # Unit tests for processing.py
  test_converting_json.py   # Unit tests for converting_json.py
  test_get_json.py          # Tests for JSON fetching
  test_json_to_raw.py       # Tests for JSON to raw conversion
  test_train.py             # Tests for model training
  json_text.py              # Test helper
  text_temp.txt             # Test fixture
  functional_test.json      # Functional test cases (JSON format)
  integration_test_cases.json  # Integration test cases (JSON format)
  json_test/
    Ans/                    # Expected outputs (7 JSON files: JOB-2018/2019-*)
    Sol/                    # Solution objects (7 .obj files matching Ans/)
```

---

## 10. ASSET INVENTORY

### Images
| Asset | Path | Description |
|-------|------|-------------|
| favicon.ico | `frontend/public/favicon.ico` | Multi-size icon (64x64, 32x32, 24x24, 16x16) |
| logo192.png | `frontend/public/logo192.png` | **Default React atom logo** (cyan atom on dark gray) — NOT a custom logo |
| logo512.png | Referenced in manifest only | Not present in repo |

### Tokenizer Files
| File | Path |
|------|------|
| vocab.txt | `artifacts/distilbert-tokenizer-pretrained-7epoch/vocab.txt` |
| tokenizer_config.json | `artifacts/distilbert-tokenizer-pretrained-7epoch/tokenizer_config.json` |
| special_tokens_map.json | `artifacts/distilbert-tokenizer-pretrained-7epoch/special_tokens_map.json` |

### Data Files (gitignored)
- `/Data` — training datasets (not in repo)
- `/Models` — model weights (not in repo, except tokenizer config)
- `*.pickle` — serialized model files (gitignored)

---

## 11. ENVIRONMENT AND TOOLING

| Setting | Value |
|---------|-------|
| Python Version | 3.9.13 |
| Node.js Version | 16.x |
| spaCy Model | `en_core_web_lg` 3.1.0 |
| PyTorch | (version varies by deployment) |
| Transformers | (version varies by deployment) |
| Package Manager | pip (Python), npm (Node.js) |
| Linting | ESLint (react-app config) |
| Testing (Python) | pytest |
| Testing (JS) | Jest + React Testing Library |
| IDE | PyCharm (mentioned in style guide) |

### Root requirements.txt (Key Packages)
- pandas 1.3.1
- numpy 1.21.1
- spacy 3.1.1
- requests 2.26.0
- en_core_web_lg 3.1.0

### Git Conventions
- Branches: `feature-XXX_XXX_XXX`
- Commits: imperative subject + body explaining context
- PRs: squash and merge, cross-reviewed
- Notebooks: `{YYYY-MM-DD} - {name}` naming convention

---

## 12. NOTEBOOK INVENTORY

| Date | Notebook | Topic |
|------|----------|-------|
| 2021-01-15 | Assessing MRSD's model | Baseline evaluation |
| 2021-09-07 | Data Processing Development | Core data pipeline |
| 2021-09-08 | HuggingFace Test | DistilBERT exploration |
| 2021-09-13 | Hierarchical Classification Test | Architecture prototyping |
| 2021-09-17 | Dissimilarity Sampling | Data sampling strategy |
| 2021-10-01 | Model Architecture | Neural network design |
| 2021-10-02 | Test Cases for Data Processing | Processing validation |
| 2021-10-03 | Visualising SSOC 2020 embeddings in 2D space | Embedding analysis |
| 2021-10-03 | SSOC 2020 Visualised.twbx | Tableau workbook |
| 2021-10-11 | Investigating Data Distribution | Class imbalance analysis |
| 2021-10-13 | Data Augmentation | Augmentation experiments |
| 2021-10-14 | XGBoost Test | Baseline ML comparison |
| 2021-10-15 | Model Training on AWS | Cloud training setup |
| 2021-10-18 | Expanding Base Dataset | Data expansion |
| 2021-11-05 | Extracting job descriptions online | Web scraping |
| 2021-12-04 | Pulling MCF Data | MCF API integration |
| 2021-12-08 | Extracting JSON data | Data extraction |
| 2021-12-09 | Extracting data for pre-training | MLM data prep |
| 2021-12-13 | Generating train and test sets | Data splits |
| 2021-12-13 | Pretraining Language Model | MLM fine-tuning |
| 2021-12-30 | Testing Lambda API calls | Deployment testing |
| 2022-01-04 | Error Analysis | Model error investigation |
| 2022-01-05 | Developing model interpretability | Explainability |
| 2022-01-10 | Testing Lambda Deployments | Production testing |
| 2022-01-30 | Loading pretrained DistilBERT model | Model loading |
| 2022-05-07 | Pretraining of Bert | Additional pre-training |
| 2022-08-12 | Changing xslx SSOC to JSON format | Data format conversion |
| 2022-10-31 | Exploring original SSOC codes selected by Emp | Employer analysis |
| 2022-11-01 | Exploring graphs of SSOC detailed descriptions | SSOC visualization |

### Plot Files
- `NumSamples_Top10Accuracy_5DSSOC.png`
- `NumSamples_TopAccuracy_5DSSOC.png`
- `Samples_Top10Accuracy_5DSSOC.png`
- `Top10Accuracy_by_1DSSOC.png`

---

## 13. RECREATION CHECKLIST

To recreate this project from scratch, a future AI should:

1. **Frontend:**
   - `npx create-react-app dirty-frontend` with Node 16
   - Install exact dependency versions from package.json (Section 3.1)
   - Add Google Fonts Roboto link to `public/index.html`
   - Recreate component tree: `App > BrowserRouter > Home > {SearchSection, LoadingSection, ResultsSection}`
   - Use MUI v5 components with `makeStyles` from `@material-ui/core` for styling
   - Apply exact color palette, spacing, and layout values from Sections 3.3-3.10
   - Wire up axios API call to prediction endpoint

2. **Backend:**
   - FastAPI app with `/predict` GET endpoint
   - MCF URL UUID extraction via regex
   - MCF API v2 integration
   - Model loading at startup (pickle for model, pretrained tokenizer directory)
   - SSOC description lookup from JSON
   - "Feeling lucky" mode with dummy data
   - Dockerize for AWS deployment

3. **ML Model:**
   - Fine-tune DistilBERT with MLM on MCF job descriptions (7 epochs)
   - Build `HierarchicalSSOCClassifier` with 5 cascading stacks
   - Use exact stack architectures from Section 5.3
   - Train with SSOC 2020 encoding (997 5-digit codes)
   - Apply data augmentation (6 techniques) during training

4. **Data Processing:**
   - Implement 3-tier HTML extraction: li tags -> p lists -> p tags
   - spaCy verb detection with override lists
   - 20+ step regex cleaning pipeline
   - MCF JSON extraction with weekly grouping

5. **Deploy:**
   - AWS Amplify backend with Docker container
   - Lambda functions as alternative
   - Heroku CORS proxy for frontend-to-API communication
