import React from 'react'

import Accordion from '@mui/material/Accordion';
import AccordionDetails from '@mui/material/AccordionDetails';
import AccordionSummary from '@mui/material/AccordionSummary';
import Typography from '@mui/material/Typography';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';


const AccordianResult = ({ useStyles, results, prob }) => {

    const classes = useStyles();

    const [expanded, setExpanded] = React.useState(false);

    const handleChange = (panel) => (event, isExpanded) => {
      setExpanded(isExpanded ? panel : false);
    };

    return (
        <Accordion expanded={expanded === 'panel1'} onChange={handleChange('panel1')}>
        <AccordionSummary
        expandIcon={<ExpandMoreIcon />}
        aria-controls="panel1bh-content"
        id="panel1bh-header"
        >
  
        <Typography className = {classes.accordianHeading}>{results[0]}</Typography>

        <Typography className = {classes.accordianSecondaryHeading}> {results[1]} </Typography>

        </AccordionSummary>
        
        <AccordionDetails>
        <Typography>
            Predicted with {prob}. Etc etc etc what ever extra stuff you want
        </Typography>
        </AccordionDetails>
        
        </Accordion>
    )
}

export default AccordianResult
