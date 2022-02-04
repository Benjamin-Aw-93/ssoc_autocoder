/*Importing the necessary packages*/
import React from 'react'
import { Accordion } from '@mui/material';
import { AccordionDetails } from '@mui/material';
import { AccordionSummary } from '@mui/material';
import { Typography } from '@mui/material';
import { Divider } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { makeStyles } from '@material-ui/core';

/*Accordian used to manage result tabs, each result will be placed in an Accordian Result component*/

const AccordianResult = ({ results, idx }) => {

    /*Style Guide*/
  const useStyles = makeStyles(theme => ({
    root: {
        backgroundColor: '#fafafa'
    },
    accordianHeading: {
        fontSize: theme.typography.pxToRem(15),
        flexBasis: '10%',
        flexShrink: 0, 
        textAlign: 'left',
        flexGrow: 0,
    },
    accordianSecondaryHeading: {
        fontSize: theme.typography.pxToRem(15),
        textAlign: 'left',
    },
    accordianThirdHeading: {
        fontSize: theme.typography.pxToRem(15),
        color: theme.palette.text.secondary,
        textAlign: 'center',
        flexBasis: '20%',
        flexShrink: 0, 
        flexGrow: 0,
    },
    }));

    const classes = useStyles();

    const [expanded, setExpanded] = React.useState(false);

    const handleChange = (panel) => (event, isExpanded) => {
      setExpanded(isExpanded ? panel : false);
    };

    return (
        /* Keeps track of each individual accordian, react will know which to open or close when clicking the arrow */
        <Accordion className={classes.root} expanded={expanded === 'panel1'} onChange={handleChange('panel1')}>
        <AccordionSummary
        expandIcon={<ExpandMoreIcon />}
        aria-controls="panel1bh-content"
        id="panel1bh-header"
        >
        {/* Header of accodian this will contain the high level summary of the results */}
        <Typography className = {classes.accordianHeading}>{idx + 2}</Typography>
        <div style={{overflow: "hidden", textOverflow: "ellipsis", width: '15rem'}}> 
        <Typography noWrap className = {classes.accordianSecondaryHeading}> {results.SSOC_Title} </Typography>
        </div>
        <Typography className = {classes.accordianThirdHeading}> [{results.Prediction_Confidence}] </Typography>

        </AccordionSummary>
        {/* Detailed results displayed with unfolded */}
        <AccordionDetails>
        <Typography><strong>SSOC Title:</strong> {results.SSOC_Title}</Typography>
        <Typography><strong>SSOC Code:</strong> {results.SSOC_Code}</Typography>
        <Typography><strong>Confidence:</strong> {results.Prediction_Confidence}</Typography>
        <Divider variant="middle"  sx={{ my: 2 }}/>
        <Typography><strong>SSOC Description:</strong></Typography>
        <Typography>{results.SSOC_Description}</Typography>
        </AccordionDetails>
        
        </Accordion>
    )
}

export default AccordianResult
