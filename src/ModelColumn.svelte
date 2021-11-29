<script> 
    import ChooseModelButton from "./ChooseModelButton.svelte";
    export let modelName; 
    export let id; 
    export let buttonText; 
</script>

<div class="divvy"> 
    <h2> {modelName} </h2>
    <ChooseModelButton id={id} text={buttonText}></ChooseModelButton>
</div>

<style> 
    *{
    margin-top: 0; 
    }
</style>