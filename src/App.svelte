<script>
  // function whenever new video is requested 

  const update = () => {
    let video = document.createElement("video");
    video.src = `https://young-horse-68.loca.lt/getVideo/20/${new Date().getTime()}`
    video.style.width = "20rem";
    video.controls = true; 
    for (let i =0; i < document.getElementById("dank").childNodes.length; i++) {
      if (document.getElementById("dank").childNodes[i].tagName == "VIDEO") {
        document.getElementById("dank").removeChild(document.getElementById("dank").childNodes[i]);
      }
    }

    document.getElementById("dank").appendChild(video);
  }
</script>


<head> 
  <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
  <meta http-equiv="Pragma" content="no-cache" />
  <meta http-equiv="Expires" content="0" />
  <div class = "content"> 
</head> 

<div class = "content" id = "dank">
  <h1> Face Morph "Interpolation" </h1>
  <p> Below is a video of an <em> interpolation </em> between generated faces (by DCGAN). 
    It basically shows what happens as the latent space vector of the original image goes in the direction of the second image. </p>
    
  <h4> Click the button to get a new interpolation! </h4>
  <button type="button" on:click={update}> Get New Face Morph </button>
  <br> 
  


</div>  


<style>

  .content {
    text-align: center;
    margin-right: 20rem; 
    margin-left: 20rem; 
  }

  p {
    font-size: 1.25rem;
  }

  h4 {
    font-size: 1.5rem;
  }

</style>
