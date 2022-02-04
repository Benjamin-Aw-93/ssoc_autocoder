import React from 'react'
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import Typography from '@material-ui/core/Typography';
import CardActions from '@material-ui/core/CardActions';
import Button from '@material-ui/core/Button';
import { Divider } from '@material-ui/core';
import CardHeader from '@material-ui/core/CardHeader';

import Box from '@mui/material/Box';

const ResultCard = ({useStyles, mainResult, originalText}) => {

    const classes = useStyles();

    return (
        originalText ? (
            <Card className={classes.root}>
            <CardHeader title="Orginal Text" className={classes.header} />
            <Divider variant="middle" />
            <CardContent>
            <Typography noWrap variant="h4" align="center">
                {mainResult.correct_ssoc ? (mainResult.mcf_job_title) : 'xxx'}
            </Typography>
            <Box component="div" sx={{ maxHeight: '165px', overflow: 'auto'}}>
            <Typography align="center">{mainResult.correct_ssoc ? mainResult.mcf_job_desc : 'xxx'} </Typography>
            </Box>
            </CardContent>
            <Divider variant="middle" />
            <CardActions className={classes.action}>
            <Button variant="contained" color="primary" className={classes.button}>
                Visit MCF Job Portal here
            </Button>
            </CardActions>
            </Card>

        ) : (
            <Card className={classes.root}>
            <CardHeader title="Result" className={classes.header} />
            <Divider variant="middle" />
            <CardContent>
            <Typography variant="h4" align="center">
                SSOC {mainResult.correct_ssoc ? (mainResult.correct_ssoc[0]) : 'xxx'}
            </Typography>
            <div className={classes.list}>
            <Typography align="center">We predict that the most probable 5D SSOC is </Typography>
            <Typography align="center"><strong>{mainResult.correct_ssoc ? mainResult.correct_ssoc[0] : 'xxx'}</strong></Typography>
            <Typography align="center">with the corresponding job title as </Typography>
            <Typography align="center"><strong>{mainResult.correct_ssoc ? mainResult.correct_ssoc[1] : 'xxx'}</strong></Typography>
            <Typography align="center">with <strong>{mainResult.correct_ssoc_proba}</strong> confidence</Typography>
            </div>
            </CardContent>
            <Divider variant="middle" />
            <CardActions className={classes.action}>
            <Button variant="contained" color="primary" className={classes.button}>
                Visit MCF Job Portal here
            </Button>
            </CardActions>
            </Card>
        )
    )
}

export default ResultCard
