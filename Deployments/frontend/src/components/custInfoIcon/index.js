/* Importing packages */
import * as React from 'react';
import { Popper } from '@mui/material';
import { Fade } from '@mui/material';
import { Paper } from '@mui/material';
import { IconButton } from '@mui/material';
import { Box } from '@mui/material';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import CancelOutlinedIcon from '@mui/icons-material/CancelOutlined';
import { makeStyles } from '@material-ui/core';


/* This Custom Info Icon is an icon that upon hover will open up a playable gif*/
const CustInfoIcon = () => {

  /* Style Guides */
  const useStyles = makeStyles(theme => ({
    [theme.breakpoints.down('md')]: {
      photo: {
        height: '320px',
        width: '144px',
      },
    },
    [theme.breakpoints.up('md')]: {
      photo: {
        height: '640px',
        width: '288px',
      },
    },
    }));
    

  const classes = useStyles();

  const [anchorEl, setAnchorEl] = React.useState(null);
  const [open, setOpen] = React.useState(false);
  const [placement, setPlacement] = React.useState();

  const handleClick = (newPlacement) => (event) => {
    setAnchorEl(event.currentTarget);
    setOpen((prev) => placement !== newPlacement || !prev);
    setPlacement(newPlacement);
  };

  return (
      <>
      {/* Popper allows us to position where the pop up box will appear */}
      <Popper open={open} anchorEl={anchorEl} placement={placement} transition>
        {({ TransitionProps }) => (
          <Fade {...TransitionProps} timeout={350}>
            <Paper>
              {/* Gif that would guide customers on how to actually use the application */}
              <img  className={classes.photo} src={require("../../imges/howtomobile.gif")} />
            </Paper>
          </Fade>
        )}
      </Popper>
        {/* IconButton which switches between two forms, open and close  */}
        <IconButton onClick={handleClick('right')}>
            {open ? <CancelOutlinedIcon fontSize="small"/> : <InfoOutlinedIcon fontSize="small"/>
            }
        </IconButton>
      </>
  );
}

export default CustInfoIcon

