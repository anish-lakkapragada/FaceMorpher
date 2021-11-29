<script> 
    import {onMount} from "svelte";
    import {MDCRipple} from '@material/ripple';

    export let id; 
    export let text; 
    export let onClickFunction = () => {}; 

    onMount(() => {
        console.log("this is id : " + id);
        const buttonRipple = new MDCRipple(document.getElementById(id))
    });

</script>

<button class="mdc-button mdc-button--outlined" {id} on:click={onClickFunction}>
    <span class="mdc-button__ripple"></span>
    <span class="mdc-button__label"> {text} </span>
</button>

<style> 
*{
    margin-top: 0; 
    column-span: all;
}
</style>
