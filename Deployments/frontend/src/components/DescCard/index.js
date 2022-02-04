/* Importing packages */
import ReactHtmlParser from 'react-html-parser';  
import { styled } from '@mui/material/styles';
import React from 'react'
import { Card } from '@mui/material';
import { CardContent }from '@mui/material';
import { Typography } from '@mui/material';
import { Divider } from '@mui/material';
import { CardHeader } from '@mui/material';
import { makeStyles } from '@material-ui/core';
import { CardActions } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { Collapse } from '@mui/material';
import { IconButton } from '@mui/material';

/* Temp component to display job description, quick fix */
const ExpandMore = styled((props) => {
    const { expand, ...other } = props;
    return <IconButton {...other} />;
  })(({ theme, expand }) => ({
    transform: !expand ? 'rotate(0deg)' : 'rotate(180deg)',
    textAlign: 'center',
    marginLeft: 'auto',
    transition: theme.transitions.create('transform', {
      duration: theme.transitions.duration.shortest,
    }),
  }));

/* This DescCard component shows uses the job desciption found on the MCF site */
const DescCard = ({mainResult}) => {

    /* Style Guides */
    const useStyles = makeStyles(theme => ({
    root: {
        borderRadius: 12,
        minWidth: 256,
        textAlign: 'center',
        backgroundColor: '#fafafa'
    },
    header: {
        textAlign: 'center',
        spacing: 10,
    },
    list: {
        padding: '20px',
    },
    action: {
        display: 'flex',
        justifyContent: 'space-around',
    },
    dropdown: {
        fontSize: theme.typography.pxToRem(15),
        color: theme.palette.text.secondary,
        marginLeft: 'auto',
    }
    }));

    const classes = useStyles();

    const [expanded, setExpanded] = React.useState(false);

    const handleExpandClick = () => {
      setExpanded(!expanded);
    };
    
    return (
            <Card className={classes.root}>
            <CardHeader title={<strong>Job Description</strong>} className={classes.header} />
            <Divider variant="middle" />
            <CardContent>
            <Typography noWrap variant="h8" align="left">
                <strong>Job Title:</strong> {mainResult.mcf_job_title ? (mainResult.mcf_job_title) : 'xxx'}
            </Typography> 
            </CardContent>
            {/* Allowing to users to expand to see more details,  */}
            <CardActions disableSpacing>
            {expanded ?  
            <Typography className={classes.dropdown}>Hide Job Description</Typography> :
            <Typography className={classes.dropdown}>Expand to show Job Description</Typography>
            }
            <ExpandMore
            expand={expanded}
            onClick={handleExpandClick}
            aria-expanded={expanded}
            aria-label="show more"
            >
                <ExpandMoreIcon />
            </ExpandMore>
            </CardActions>
            <Collapse in={expanded} timeout="auto" unmountOnExit>
            <CardContent>
            <Typography paragraph>
            <Typography align="left">{mainResult.mcf_job_desc ? ReactHtmlParser(mainResult.mcf_job_desc) : 'xxx'} </Typography>
            </Typography>
            </CardContent>
            </Collapse>
            </Card>
    )
}

export default DescCard
